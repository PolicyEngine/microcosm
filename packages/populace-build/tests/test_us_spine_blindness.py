"""Tripwire against accidental spine-awareness, not an adversarial sandbox.

This guard enforces, and its tests certify, exactly these surfaces:

- direct attribute, subscript, ``.loc``, and ``__getitem__`` reads;
- canonical guarded-column factory calls, including aliases bound by simple
  assignment or named expression;
- ``query``, ``eval``, ``filter``, and ``get`` expression surfaces, failing
  closed on opacity including hidden arguments and method aliases bound by
  simple assignment or named expression;
- one-level static indirection through constants, concatenation, f-strings,
  ``str.format`` with full field syntax, ``%`` formatting, ``str * int``,
  static ``.replace`` chains, and the static-receiver case methods ``lower``,
  ``upper``, ``casefold``, ``title``, and ``capitalize``;
- loop and comprehension propagation over static iterables; and
- contraband guarded-name literals anywhere statically visible in non-owner
  modules.

Two classes are out of scope by design, and naming them is the honest
boundary. First, deliberately obfuscated construction -- reverse slicing,
``format_map`` over dynamic maps, ``__doc__`` or ``__annotations__`` mining,
container-indexed method aliases, and kin -- is controlled by code review and
the adversarial merge-review process. A scanner that claimed to catch code
written to deceive would be lying. Second, column names materialized purely
from runtime data are controlled by the assembly receipt and runtime
validation.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from itertools import product
from pathlib import Path
from string import Formatter

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_US_RUNTIME = (
    Path(__file__).resolve().parents[1] / "src" / "populace" / "build" / "us_runtime"
)
_US_RUNTIME_IMPORT_PREFIX = "populace.build.us_runtime"
_SPINE_BLIND_BUILD_TOOLS = (_REPOSITORY_ROOT / "tools" / "build_us_multispine_pool.py",)
_REQUIRED_POOL_RUNTIME_MODULES = frozenset(
    {
        "multispine_pool.py",
        "puf_support.py",
        "spine_agreement.py",
        "spine_assembly.py",
    }
)
_RETIRED_LATE_ASSEMBLY_MODULES = frozenset(
    {
        "acs_multispine.py",
        "base_pool.py",
    }
)

# These modules own source-spine provenance rather than applying population
# treatments. Keep the allowlist exact so adding a new exception requires a
# reviewed contract change.
_SOURCE_SPINE_PROVENANCE_OWNERS = frozenset(
    {
        "base_pool.py",  # Legacy late-spine assembly.
        # Enumerates provenance columns only to reject preassembled source frames.
        "operator_boundary.py",
        "puf_qrf_chain.py",  # Carries provenance into resumable checkpoints.
        "puf_support.py",  # Validates provenance at the clone boundary.
        "spine_agreement.py",  # Pre-calibration distribution comparison.
        "spine_assembly.py",  # New pre-operator assembly seam.
        "support_provenance.py",  # Centralized provenance compatibility.
        "warm_start_selection.py",  # Provenance reporting and recovery.
    }
)

_US_ENTITIES = (
    "benefit_unit",
    "benunit",
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
    for suffix in ("spine", "spine_source_id", "support_channel")
)
_SOURCE_SPINE_COLUMN_FACTORIES = frozenset(
    {
        "spine_source_id_column",
        "spine_column",
        "support_channel_column",
    }
)
_STRICT_COLUMN_METHODS = frozenset({"query", "eval", "filter", "get"})
_PUF_CLONE_OPERATOR_MODULES = (
    "puf_capital_gains_tail.py",
    "puf_qrf_chain.py",
    "puf_support.py",
)
_SPINE_BLIND_OPERATOR_MODULES = (
    "acs_transfer.py",
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
    "multispine_pool.py",
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

# Every runtime module must be deliberately classified. This allowlist does
# not exempt a module from the all-runtime AST scan below; it only records
# modules outside the migrated population-treatment registry. Keeping the
# inventory explicit makes an added us_runtime module fail until reviewed.
_OTHER_US_RUNTIME_MODULES = frozenset(
    {
        "__init__.py",
        "acs_inputs.py",
        "acs_multispine.py",
        "acs_pums.py",
        "acs_sources.py",
        "asec_checkpoint.py",  # Bounded checkpoint I/O; no population treatment.
        "asec_pool.py",
        "base_pool.py",
        "block_ladder_sources.py",
        "capital_gain_distributions.py",
        "casualty_losses.py",
        "congressional_district_geography.py",
        "congressional_district_vintage.py",
        "congressional_district_vintage_crosswalk.py",
        "cps_carried.py",
        "demographics.py",
        "education_assistance_source.py",
        "eligibility_inputs.py",
        "engine_lifecycle.py",
        "fiscal_targets.py",
        "geography_ladder.py",
        "hours_worked.py",
        "h5_io.py",  # US artifact I/O; no population treatment.
        "immigration.py",
        "input_mass.py",
        "l0_refit_export.py",
        "medicaid_take_up.py",
        "misc_itemized.py",
        "nonzero_shares.py",
        "operator_boundary.py",  # Raw-stage validator; no population treatment.
        "org_wages.py",
        "parity_reference.py",
        "pregnancy.py",
        "puf_aggregate_records.py",
        "puf_capital_gains_tail.py",
        "puf_donor_io.py",  # Bounded donor artifact I/O; no population treatment.
        "puf_e01000_reconciliation.py",
        "puf_interest_components.py",
        "puf_qrf_chain.py",
        "puf_qrf_worker.py",
        "puf_source_agi.py",
        "puf_support.py",
        "puma_ladder.py",
        "puma_ladder_sources.py",
        "reform_coverage_smoke.py",
        "reform_validation.py",
        "register_consistency.py",
        "relationship_inputs.py",
        "release_gate_preflight.py",
        "release_input_coverage.py",
        "release_target_parity.py",
        "scf_auto_loans.py",
        "scf_wealth.py",
        "sipp_financial_assets.py",
        "sipp_tips.py",
        "sipp_vehicles.py",
        "snap_discretionary_exemption.py",
        "snap_state_take_up.py",
        "snap_take_up.py",
        "source_coverage.py",
        "source_runtime.py",
        "sources.py",
        "spine_agreement.py",
        "spine_assembly.py",
        "spm_resources.py",
        "support_provenance.py",
        "take_up.py",
        "take_up_contract.py",
        "target_aging.py",
        "validation_input_coverage.py",
        "warm_start_selection.py",
    }
)
_CLASSIFIED_US_RUNTIME_MODULES = frozenset(_SPINE_BLIND_OPERATOR_MODULES).union(
    _OTHER_US_RUNTIME_MODULES
)


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.NamedExpr):
        return function.target.id
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _string_shape(node: ast.AST) -> str | None:
    """Resolve a static string, using ``*`` for a dynamic formatted value."""

    literal = _literal_string(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.FormattedValue):
        return "*"
    if isinstance(node, ast.JoinedStr):
        pieces = [_string_shape(value) for value in node.values]
        if any(piece is None for piece in pieces):
            return None
        return "".join(piece for piece in pieces if piece is not None)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_shape(node.left)
        right = _string_shape(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr == "join" and len(node.args) == 1:
        separator = _string_shape(node.func.value)
        values = node.args[0]
        if separator is None or not isinstance(values, (ast.List, ast.Tuple)):
            return None
        pieces = [_string_shape(value) for value in values.elts]
        if any(piece is None for piece in pieces):
            return None
        return separator.join(piece for piece in pieces if piece is not None)
    if node.func.attr == "format":
        template = _string_shape(node.func.value)
        if template is None or node.keywords:
            return None
        for argument in node.args:
            value = _string_shape(argument)
            template = template.replace(
                "{}",
                "*" if value is None else value,
                1,
            )
        return template
    return None


def _is_source_column_shape(shape: str) -> bool:
    if not any(
        marker in shape for marker in ("_spine", "_spine_source_id", "_support_channel")
    ):
        return False
    return any(
        fnmatch.fnmatchcase(column, shape) for column in _OPERATOR_SOURCE_COLUMNS
    )


def _factory_aliases(tree: ast.AST) -> set[str]:
    aliases = set(_SOURCE_SPINE_COLUMN_FACTORIES)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported in node.names:
            if imported.name in _SOURCE_SPINE_COLUMN_FACTORIES:
                aliases.add(imported.asname or imported.name)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.NamedExpr):
                value = node.value
                targets = [node.target]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            else:
                continue
            if isinstance(value, ast.Name):
                if value.id not in aliases:
                    continue
            elif isinstance(value, ast.Attribute):
                # provenance.support_channel_column — module-qualified
                # references to canonical factories are ordinary imports,
                # not obfuscation (sol #583 round 6).
                if value.attr not in _SOURCE_SPINE_COLUMN_FACTORIES:
                    continue
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    return aliases


def _source_expression(
    node: ast.AST,
    *,
    bindings: list[dict[str, str | None]],
    factory_aliases: set[str],
) -> str | None:
    if isinstance(node, ast.Name):
        for scope in reversed(bindings):
            if node.id in scope:
                return scope[node.id]
        return None
    if isinstance(node, ast.Call) and _call_name(node) in factory_aliases:
        return f"call to {_call_name(node)}"
    shape = _string_shape(node)
    if shape is not None and _is_source_column_shape(shape):
        return f"source column {shape!r}"
    return None


def _subscript_source_expression(
    node: ast.AST,
    *,
    bindings: list[dict[str, str | None]],
    factory_aliases: set[str],
) -> str | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            description = _subscript_source_expression(
                element,
                bindings=bindings,
                factory_aliases=factory_aliases,
            )
            if description is not None:
                return description
        return None
    return _source_expression(
        node,
        bindings=bindings,
        factory_aliases=factory_aliases,
    )


def _static_string_shape(
    node: ast.AST, constants: list[dict[str, object]]
) -> str | None:
    """_string_shape with one level of static constant propagation.

    Names bound to resolvable string literals in an enclosing scope resolve
    to their value (including inside f-string interpolations), so
    ``expr = '...'; df.query(expr)`` and ``df.query(f"{col} == 'x'")`` are
    seen through. Anything still opaque returns None — and the caller
    treats an opaque pandas expression as a violation (fail-closed).
    """

    if isinstance(node, ast.Name):
        for scope in reversed(constants):
            if node.id in scope:
                value = scope[node.id]
                return value if isinstance(value, str) else None
        return None
    if isinstance(node, ast.NamedExpr):
        return _static_string_shape(node.value, constants)
    if isinstance(node, ast.IfExp):
        return None
    if isinstance(node, ast.FormattedValue):
        inner = _static_string_shape(node.value, constants)
        return inner if inner is not None else _OPAQUE_STRING_PART
    if isinstance(node, ast.JoinedStr):
        pieces = [_static_string_shape(value, constants) for value in node.values]
        if any(piece is None for piece in pieces):
            return None
        return "".join(piece for piece in pieces if piece is not None)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left = _static_string_shape(node.left, constants)
            right = _static_string_shape(node.right, constants)
            if left is not None and right is not None:
                return left + right
            return None
        if isinstance(node.op, ast.Mult):
            left_string = _static_string_shape(node.left, constants)
            right_string = _static_string_shape(node.right, constants)
            left_integer = _static_integer(node.left, constants)
            right_integer = _static_integer(node.right, constants)
            if left_string is not None and right_integer is not None:
                return left_string * right_integer
            if left_integer is not None and right_string is not None:
                return left_integer * right_string
            return None
        if isinstance(node.op, ast.Mod):
            template = _static_string_shape(node.left, constants)
            operand = _static_percent_operand(node.right, constants)
            if template is None or operand is _OPAQUE_STATIC_VALUE:
                return None
            try:
                result = template % operand
            except (TypeError, ValueError):
                return None
            return result if isinstance(result, str) else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"lower", "upper", "casefold", "title", "capitalize"}
        and not node.args
        and not node.keywords
    ):
        value = _static_string_shape(node.func.value, constants)
        if value is None:
            return None
        case_method = {
            "lower": str.lower,
            "upper": str.upper,
            "casefold": str.casefold,
            "title": str.title,
            "capitalize": str.capitalize,
        }[node.func.attr]
        return case_method(value)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and not node.keywords
    ):
        separator = _static_string_shape(node.func.value, constants)
        values = _static_string_list(node.args[0], constants)
        if separator is None or values is None:
            return None
        return separator.join(values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and not node.keywords
        and len(node.args) in {2, 3}
    ):
        value = _static_string_shape(node.func.value, constants)
        old = _static_string_shape(node.args[0], constants)
        new = _static_string_shape(node.args[1], constants)
        count = (
            _static_integer(node.args[2], constants) if len(node.args) == 3 else None
        )
        if value is None or old is None or new is None:
            return None
        if len(node.args) == 3 and count is None:
            return None
        return (
            value.replace(old, new) if count is None else value.replace(old, new, count)
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        template = _static_string_shape(node.func.value, constants)
        if template is None:
            return None
        return _resolve_static_format(template, node, constants)
    return _string_shape(node)


_OPAQUE_STATIC_VALUE = object()
_OPAQUE_STRING_PART = "\N{OBJECT REPLACEMENT CHARACTER}"
_OPAQUE_METHOD_ALIAS = ("", True)


class _StaticStringChoices(tuple):
    """Abstract alternatives bound by one static loop/comprehension target."""


def _static_format_value(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> object:
    value = _static_string_shape(node, constants)
    if value is not None:
        return _OPAQUE_STATIC_VALUE if _OPAQUE_STRING_PART in value else value
    integer = _static_integer(node, constants)
    if integer is not None:
        return integer
    return _static_literal_value(node, constants)


def _static_literal_value(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> object:
    if isinstance(node, ast.Name):
        for scope in reversed(constants):
            if node.id in scope:
                value = scope[node.id]
                return _OPAQUE_STATIC_VALUE if value is None else value
        return _OPAQUE_STATIC_VALUE
    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        (str, int, float, bool, bytes, type(None)),
    ):
        return node.value
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values = tuple(
            _static_literal_value(element, constants) for element in node.elts
        )
        if any(value is _OPAQUE_STATIC_VALUE for value in values):
            return _OPAQUE_STATIC_VALUE
        if isinstance(node, ast.List):
            return list(values)
        if isinstance(node, ast.Set):
            try:
                return set(values)
            except TypeError:
                return _OPAQUE_STATIC_VALUE
        return values
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            return _OPAQUE_STATIC_VALUE
        keys = tuple(
            _static_literal_value(key, constants)
            for key in node.keys
            if key is not None
        )
        values = tuple(_static_literal_value(value, constants) for value in node.values)
        if any(value is _OPAQUE_STATIC_VALUE for value in (*keys, *values)):
            return _OPAQUE_STATIC_VALUE
        try:
            return dict(zip(keys, values, strict=True))
        except TypeError:
            return _OPAQUE_STATIC_VALUE
    return _OPAQUE_STATIC_VALUE


def _resolve_static_format(
    template: str,
    node: ast.Call,
    constants: list[dict[str, object]],
) -> str:
    """Substitute every statically known ``str.format`` field."""

    formatter = Formatter()
    positional: list[object] = []
    expanded_positional = False
    for argument in node.args:
        if not isinstance(argument, ast.Starred):
            positional.append(_static_format_value(argument, constants))
            continue
        expanded = _static_literal_value(argument.value, constants)
        if isinstance(expanded, (list, tuple)):
            positional.extend(expanded)
        else:
            expanded_positional = True

    keywords: dict[str, object] = {}
    expanded_keywords = False
    for keyword in node.keywords:
        if keyword.arg is not None:
            keywords[keyword.arg] = _static_format_value(keyword.value, constants)
            continue
        expanded = _static_literal_value(keyword.value, constants)
        if isinstance(expanded, dict) and all(isinstance(key, str) for key in expanded):
            keywords.update(expanded)
        else:
            expanded_keywords = True

    auto_index = [0]

    def resolve_fields(value: str) -> str:
        pieces: list[str] = []
        try:
            parsed = tuple(formatter.parse(value))
        except ValueError:
            return _OPAQUE_STRING_PART
        for literal, field_name, format_spec, conversion in parsed:
            pieces.append(literal)
            if field_name is None:
                continue
            lookup_name = field_name
            if field_name == "":
                lookup_name = str(auto_index[0])
                auto_index[0] += 1
            root_name = re.split(r"[.[]", lookup_name, maxsplit=1)[0]
            if (
                expanded_positional
                and root_name.isdecimal()
                or expanded_keywords
                and not root_name.isdecimal()
                and root_name not in keywords
            ):
                pieces.append(_OPAQUE_STRING_PART)
                continue
            try:
                field_value, _ = formatter.get_field(
                    lookup_name,
                    tuple(positional),
                    keywords,
                )
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                field_value = _OPAQUE_STATIC_VALUE
            if field_value is _OPAQUE_STATIC_VALUE:
                pieces.append(_OPAQUE_STRING_PART)
                continue
            resolved_spec = resolve_fields(format_spec) if format_spec else ""
            if _OPAQUE_STRING_PART in resolved_spec:
                pieces.append(_OPAQUE_STRING_PART)
                continue
            try:
                converted = formatter.convert_field(field_value, conversion)
                pieces.append(formatter.format_field(converted, resolved_spec))
            except (TypeError, ValueError):
                pieces.append(_OPAQUE_STRING_PART)
        return "".join(pieces)

    try:
        return resolve_fields(template)
    except (IndexError, KeyError, ValueError):
        return _OPAQUE_STRING_PART


def _static_integer(node: ast.AST, constants: list[dict[str, object]]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        for scope in reversed(constants):
            if node.id in scope:
                value = scope[node.id]
                return value if isinstance(value, int) else None
    if isinstance(node, ast.NamedExpr):
        return _static_integer(node.value, constants)
    return None


def _static_percent_operand(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> object:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bytes)
    ):
        return node.value
    if isinstance(node, ast.Name):
        for scope in reversed(constants):
            if node.id in scope:
                value = scope[node.id]
                if isinstance(value, (str, int, float, bytes, tuple, dict)):
                    return value
                return _OPAQUE_STATIC_VALUE
        return _OPAQUE_STATIC_VALUE
    if isinstance(node, ast.NamedExpr):
        return _static_percent_operand(node.value, constants)
    if isinstance(node, ast.Tuple):
        values = tuple(
            _static_percent_operand(element, constants) for element in node.elts
        )
        if any(value is _OPAQUE_STATIC_VALUE for value in values):
            return _OPAQUE_STATIC_VALUE
        return values
    if isinstance(node, ast.Dict):
        value = _static_literal_value(node, constants)
        return value if isinstance(value, dict) else _OPAQUE_STATIC_VALUE
    return _OPAQUE_STATIC_VALUE


def _static_string_list(
    node: ast.AST, constants: list[dict[str, object]]
) -> tuple[str, ...] | None:
    """Resolve any statically enumerable string iterable."""

    if isinstance(node, ast.Name):
        for scope in reversed(constants):
            if node.id in scope:
                value = scope[node.id]
                if isinstance(value, dict) and all(
                    isinstance(item, str) for item in value
                ):
                    return tuple(value)
                if isinstance(value, (list, set, tuple)) and all(
                    isinstance(item, str) for item in value
                ):
                    return tuple(value)
                if isinstance(value, str):
                    return tuple(value)
                return None
        return None
    literal = _literal_string(node)
    if literal is not None:
        return tuple(literal)
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        items: list[str] = []
        for element in node.elts:
            values = _static_string_values(element, constants)
            if values is None:
                return None
            items.extend(values)
        return tuple(items)
    if isinstance(node, ast.Dict):
        items: list[str] = []
        for key in node.keys:
            if key is None:
                return None
            values = _static_string_values(key, constants)
            if values is None:
                return None
            items.extend(values)
        return tuple(items)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string_list(node.left, constants)
        right = _static_string_list(node.right, constants)
        if left is not None and right is not None:
            return (*left, *right)
        return None
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        local: dict[str, object] = {}
        nested_constants = [*constants, local]
        for generator in node.generators:
            if generator.ifs or generator.is_async:
                return None
            values = _static_string_list(generator.iter, nested_constants)
            if values is None:
                return None
            for name in _assigned_names(generator.target):
                local[name] = _StaticStringChoices(values)
        return _static_string_values(node.elt, nested_constants)
    return None


def _active_static_string_choices(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> tuple[tuple[str, _StaticStringChoices], ...]:
    """Find loaded loop/comprehension choices used by one expression."""

    choices: list[tuple[str, _StaticStringChoices]] = []
    names = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    for name in sorted(names):
        for scope in reversed(constants):
            if name not in scope:
                continue
            value = scope[name]
            if isinstance(value, _StaticStringChoices):
                choices.append((name, value))
            break
    return tuple(choices)


def _static_string_values(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> tuple[str, ...] | None:
    active_choices = _active_static_string_choices(node, constants)
    if active_choices:
        resolved: list[str] = []
        names = tuple(name for name, _ in active_choices)
        alternatives = tuple(tuple(values) for _, values in active_choices)
        for selected in product(*alternatives):
            values = _static_string_values(
                node,
                [*constants, dict(zip(names, selected, strict=True))],
            )
            if values is None:
                return None
            resolved.extend(values)
        return tuple(dict.fromkeys(resolved))
    if isinstance(node, ast.JoinedStr):
        alternatives = _static_joined_string_values(node, constants)
        if alternatives is not None:
            return alternatives
    shape = _static_string_shape(node, constants)
    if shape is not None:
        return (shape,)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        alternatives = _static_concatenated_string_values(node, constants)
        if alternatives is not None:
            return alternatives
    return _static_string_list(node, constants)


def _static_composition_operand_values(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> tuple[str, ...] | None:
    """Resolve scalar strings or abstract iteration choices for composition."""

    if isinstance(node, ast.Name):
        for scope in reversed(constants):
            if node.id in scope:
                value = scope[node.id]
                if isinstance(value, _StaticStringChoices):
                    return tuple(value)
                break
    if isinstance(node, ast.NamedExpr):
        return _static_composition_operand_values(node.value, constants)
    shape = _static_string_shape(node, constants)
    if shape is not None:
        return (shape,)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _static_concatenated_string_values(node, constants)
    return None


def _static_concatenated_string_values(
    node: ast.BinOp,
    constants: list[dict[str, object]],
) -> tuple[str, ...] | None:
    """Expand concatenation across abstract loop/comprehension alternatives."""

    left = _static_composition_operand_values(node.left, constants)
    right = _static_composition_operand_values(node.right, constants)
    if left is None or right is None:
        return None
    return tuple(prefix + suffix for prefix in left for suffix in right)


def _static_joined_string_values(
    node: ast.JoinedStr,
    constants: list[dict[str, object]],
) -> tuple[str, ...] | None:
    """Expand every statically enumerable f-string field choice."""

    assembled = ("",)
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            choices = (part.value,)
        elif isinstance(part, ast.FormattedValue):
            composition_choices = _static_composition_operand_values(
                part.value,
                constants,
            )
            if composition_choices is not None:
                choices: tuple[object, ...] = composition_choices
            else:
                static_value = _static_format_value(part.value, constants)
                if static_value is _OPAQUE_STATIC_VALUE:
                    return None
                choices = (static_value,)
            format_spec = (
                ""
                if part.format_spec is None
                else _static_string_shape(part.format_spec, constants)
            )
            if format_spec is None or _OPAQUE_STRING_PART in format_spec:
                return None
            formatted: list[str] = []
            for choice in choices:
                if part.conversion == ord("s"):
                    converted = str(choice)
                elif part.conversion == ord("r"):
                    converted = repr(choice)
                elif part.conversion == ord("a"):
                    converted = ascii(choice)
                else:
                    converted = choice
                try:
                    formatted.append(format(converted, format_spec))
                except (TypeError, ValueError):
                    return None
            choices = tuple(formatted)
        else:
            choices = _static_string_values(part, constants)
            if choices is None:
                return None
        assembled = tuple(prefix + choice for prefix in assembled for choice in choices)
    return assembled


def _statically_visible_source_columns(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> tuple[str, ...]:
    """Return guarded names present in a statically resolvable expression."""

    values = _static_string_values(node, constants)
    if values is None:
        return ()
    return tuple(
        column
        for column in sorted(_OPERATOR_SOURCE_COLUMNS)
        if any(column in value for value in values)
    )


def _docstring_value_ids(tree: ast.AST) -> set[int]:
    """Identify only true scope docstrings, not arbitrary string expressions."""

    scope_nodes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings: set[int] = set()
    for scope in ast.walk(tree):
        if not isinstance(scope, scope_nodes) or not scope.body:
            continue
        statement = scope.body[0]
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            docstrings.add(id(statement.value))
    return docstrings


def _pandas_expression_source(node: ast.AST) -> str | None:
    """Resolve guarded column-shaped tokens in a static pandas expression."""

    shape = _string_shape(node)
    if shape is None:
        return None
    for token in re.findall(r"[\w*?\[\]-]+", shape):
        if _is_source_column_shape(token):
            return f"source column {token!r}"
    return None


def _call_argument(
    node: ast.Call,
    *,
    position: int,
    keyword: str,
) -> ast.AST | None:
    if len(node.args) > position:
        return node.args[position]
    return next(
        (candidate.value for candidate in node.keywords if candidate.arg == keyword),
        None,
    )


def _assigned_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.List, ast.Tuple)):
        return tuple(
            name for element in target.elts for name in _assigned_names(element)
        )
    return ()


class _ScopeAssignmentCounter(ast.NodeVisitor):
    """Count binding sites in one lexical scope, excluding nested scopes."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.external_names: set[str] = set()

    def _count(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._count(node.id)

    def _visit_defaults_and_decorators(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self._count(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_defaults_and_decorators(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_defaults_and_decorators(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._count(node.name)
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            self._count(imported.asname or imported.name.split(".", maxsplit=1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for imported in node.names:
            self._count(imported.asname or imported.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self._count(node.name)
        if node.type is not None:
            self.visit(node.type)
        for statement in node.body:
            self.visit(statement)

    def visit_Global(self, node: ast.Global) -> None:
        self.external_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.external_names.update(node.names)

    def _count_comprehension_named_expressions(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        for descendant in ast.walk(node):
            if isinstance(descendant, ast.NamedExpr):
                for name in _assigned_names(descendant.target):
                    self._count(name)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._count_comprehension_named_expressions(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._count_comprehension_named_expressions(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._count_comprehension_named_expressions(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._count_comprehension_named_expressions(node)


def _nested_external_writes(
    body: list[ast.stmt],
    declaration: type[ast.Global] | type[ast.Nonlocal],
) -> set[str]:
    class NestedWriteCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0
            self.declared: set[str] = set()
            self.stored: set[str] = set()

        def _visit_function(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def visit_Global(self, node: ast.Global) -> None:
            if self.depth and declaration is ast.Global:
                self.declared.update(node.names)

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            if self.depth and declaration is ast.Nonlocal:
                self.declared.update(node.names)

        def visit_Name(self, node: ast.Name) -> None:
            if self.depth and isinstance(node.ctx, ast.Store):
                self.stored.add(node.id)

    collector = NestedWriteCollector()
    for statement in body:
        collector.visit(statement)
    return collector.declared & collector.stored


def _scope_assignment_counts(
    body: list[ast.stmt],
    *,
    parameters: tuple[str, ...] = (),
    nested_declaration: type[ast.Global] | type[ast.Nonlocal] = ast.Nonlocal,
) -> dict[str, int]:
    counter = _ScopeAssignmentCounter()
    for statement in body:
        counter.visit(statement)
    for name in counter.external_names:
        counter.counts.pop(name, None)
    for name in _nested_external_writes(body, nested_declaration):
        counter._count(name)
    for name in parameters:
        counter._count(name)
    return counter.counts


class _SourceReadVisitor(ast.NodeVisitor):
    def __init__(
        self,
        factory_aliases: set[str],
        *,
        docstring_value_ids: set[int],
    ) -> None:
        self.factory_aliases = factory_aliases
        self.docstring_value_ids = docstring_value_ids
        self.bindings: list[dict[str, str | None]] = [{}]
        self.constants: list[dict[str, object]] = [{}]
        self.column_containers: list[dict[str, bool]] = [{}]
        self.attribute_containers: list[dict[str, bool]] = [{}]
        self.method_aliases: list[dict[str, tuple[str, bool] | None]] = [{}]
        self.method_alias_history: list[set[str]] = [set()]
        self.assignment_counts: list[dict[str, int]] = [{}]
        self.scope_kinds = ["module"]
        self.class_lexical_flow_states: list[
            tuple[
                dict[str, str | None],
                dict[str, object],
                dict[str, bool],
                dict[str, bool],
                dict[str, tuple[str, bool] | None],
                set[str],
            ]
        ] = []
        self.class_lexical_scope_depths: list[int] = []
        self.accesses: set[tuple[int, int, str]] = set()

    def visit(self, node: ast.AST) -> object:
        if isinstance(node, ast.expr) and id(node) not in self.docstring_value_ids:
            for column in _statically_visible_source_columns(node, self.constants):
                self._record(
                    node,
                    f"contraband source column {column!r}",
                )
        return super().visit(node)

    def _expression(self, node: ast.AST) -> str | None:
        return _source_expression(
            node,
            bindings=self.bindings,
            factory_aliases=self.factory_aliases,
        )

    def _record(self, node: ast.AST, description: str) -> None:
        self.accesses.add((node.lineno, node.col_offset, description))

    def _column_container(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            for scope in reversed(self.column_containers):
                if node.id in scope:
                    return scope[node.id]
            return False
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "copy"
        ):
            return self._column_container(node.func.value)
        return False

    def _attribute_container(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            for scope in reversed(self.attribute_containers):
                if node.id in scope:
                    return scope[node.id]
            return False
        if isinstance(node, ast.Call):
            if _call_name(node) in {"DataFrame", "table"}:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "copy":
                return self._attribute_container(node.func.value)
        return False

    def _method_alias(
        self,
        node: ast.AST,
    ) -> tuple[str, bool] | None:
        if isinstance(node, ast.NamedExpr):
            return self._method_alias(node.value)
        if isinstance(node, ast.Name):
            for scope in reversed(self.method_aliases):
                if node.id in scope:
                    return scope[node.id]
            if node.id == "getattr":
                return "getattr", True
            return None
        if isinstance(node, ast.Attribute) and node.attr in _STRICT_COLUMN_METHODS:
            strict_opacity = node.attr != "get" or self._attribute_container(node.value)
            return node.attr, strict_opacity
        if (
            isinstance(node, ast.Call)
            and _call_name(node) == "getattr"
            and len(node.args) >= 2
        ):
            attribute = _static_string_shape(node.args[1], self.constants)
            if (
                attribute in _STRICT_COLUMN_METHODS
                and _OPAQUE_STRING_PART not in attribute
            ):
                strict_opacity = attribute != "get" or self._attribute_container(
                    node.args[0]
                )
                return attribute, strict_opacity
        return None

    def _bind_name(
        self,
        name: str,
        value: ast.AST,
        *,
        scope_index: int = -1,
    ) -> None:
        description = self._expression(value)
        constant: object = _static_string_shape(value, self.constants)
        if constant is None:
            constant = _static_integer(value, self.constants)
        if constant is None:
            literal = _static_literal_value(value, self.constants)
            if literal is not _OPAQUE_STATIC_VALUE:
                constant = literal
        if constant is None:
            constant = _static_string_list(value, self.constants)
        method_alias = self._method_alias(value)
        if method_alias is None and name in self.method_alias_history[scope_index]:
            method_alias = _OPAQUE_METHOD_ALIAS

        self.bindings[scope_index][name] = description
        # None is an explicit opaque shadow: lookups must stop here,
        # never fall through to a stale outer binding (shadowed
        # parameters and conditional reassignments — sol round 3).
        self.constants[scope_index][name] = constant
        self.column_containers[scope_index][name] = self._column_container(value)
        self.attribute_containers[scope_index][name] = self._attribute_container(value)
        self.method_aliases[scope_index][name] = method_alias
        if method_alias is not None:
            self.method_alias_history[scope_index].add(name)

    def _bind_target(
        self,
        target: ast.AST,
        value: ast.AST,
        *,
        scope_index: int = -1,
    ) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, value, scope_index=scope_index)
            return
        if isinstance(target, ast.Starred):
            for name in _assigned_names(target.value):
                self._bind_name(name, value, scope_index=scope_index)
            return
        if isinstance(target, (ast.List, ast.Tuple)):
            if isinstance(value, (ast.List, ast.Tuple)) and len(target.elts) == len(
                value.elts
            ):
                for child_target, child_value in zip(
                    target.elts,
                    value.elts,
                    strict=True,
                ):
                    self._bind_target(
                        child_target,
                        child_value,
                        scope_index=scope_index,
                    )
                return
            for name in _assigned_names(target):
                self._bind_name(name, value, scope_index=scope_index)

    def _bind(
        self,
        targets: list[ast.AST],
        value: ast.AST,
        *,
        scope_index: int = -1,
    ) -> None:
        for target in targets:
            self._bind_target(target, value, scope_index=scope_index)

    def _visit_access_target(self, target: ast.AST) -> None:
        """Visit column-bearing stores without treating names as string uses."""

        if isinstance(target, (ast.Attribute, ast.Subscript)):
            self.visit(target)
        elif isinstance(target, ast.Starred):
            self._visit_access_target(target.value)
        elif isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._visit_access_target(element)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._visit_access_target(target)
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                self._poison(target.value.id)
        self._bind(list(node.targets), node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.visit(node.target)
        for name in _assigned_names(node.target):
            self._poison(name)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            self._visit_access_target(node.target)
            for name in _assigned_names(node.target):
                self.bindings[-1][name] = None
                self.constants[-1][name] = None
                self.column_containers[-1][name] = False
                self.attribute_containers[-1][name] = False
                self.method_aliases[-1][name] = None
            return
        self.visit(node.value)
        self._visit_access_target(node.target)
        self._bind([node.target], node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        scope_index = next(
            index
            for index in range(len(self.scope_kinds) - 1, -1, -1)
            if self.scope_kinds[index] != "comprehension"
        )
        self._bind([node.target], node.value, scope_index=scope_index)

    def _bind_iteration_rows(
        self,
        target: ast.AST,
        iterable: ast.AST,
    ) -> bool:
        """Bind tuple-unpacking loop targets over static rows of strings.

        ``for entity, suffix in PAIRS`` with ``PAIRS = (("person",
        "support_channel"),)`` binds ``entity``/``suffix`` to their
        per-position choice sets — natural declarative loop code, in
        scope (sol #583 round 6).
        """

        literal = _static_literal_value(iterable, self.constants)
        if literal is _OPAQUE_STATIC_VALUE or not isinstance(literal, (list, tuple)):
            return False
        rows = tuple(literal)
        if not rows or not all(
            isinstance(row, (list, tuple))
            and all(isinstance(item, str) for item in row)
            for row in rows
        ):
            return False
        width = len(rows[0])
        if any(len(row) != width for row in rows) or not (
            isinstance(target, (ast.Tuple, ast.List))
            and len(target.elts) == width
            and all(isinstance(element, ast.Name) for element in target.elts)
        ):
            return False
        for position, element in enumerate(target.elts):
            self._bind_iteration_target(
                element,
                tuple(row[position] for row in rows),
            )
        return True

    def _bind_iteration_target(
        self,
        target: ast.AST,
        values: tuple[str, ...] | None,
    ) -> None:
        for name in _assigned_names(target):
            self.bindings[-1][name] = None
            self.constants[-1][name] = (
                None if values is None else _StaticStringChoices(values)
            )
            self.column_containers[-1][name] = False
            self.attribute_containers[-1][name] = False
            self.method_aliases[-1][name] = (
                _OPAQUE_METHOD_ALIAS if name in self.method_alias_history[-1] else None
            )
            if self.scope_kinds[-1] == "comprehension":
                self.assignment_counts[-1][name] = 1

    def _flow_state(
        self,
    ) -> tuple[
        dict[str, str | None],
        dict[str, object],
        dict[str, bool],
        dict[str, bool],
        dict[str, tuple[str, bool] | None],
        set[str],
    ]:
        return (
            self.bindings[-1].copy(),
            self.constants[-1].copy(),
            self.column_containers[-1].copy(),
            self.attribute_containers[-1].copy(),
            self.method_aliases[-1].copy(),
            self.method_alias_history[-1].copy(),
        )

    def _restore_flow_state(
        self,
        state: tuple[
            dict[str, str | None],
            dict[str, object],
            dict[str, bool],
            dict[str, bool],
            dict[str, tuple[str, bool] | None],
            set[str],
        ],
    ) -> None:
        (
            self.bindings[-1],
            self.constants[-1],
            self.column_containers[-1],
            self.attribute_containers[-1],
            self.method_aliases[-1],
            self.method_alias_history[-1],
        ) = tuple(value.copy() for value in state)

    @staticmethod
    def _same_flow_value(left: object, right: object) -> bool:
        return type(left) is type(right) and left == right

    def _merge_flow_states(
        self,
        left: tuple[
            dict[str, str | None],
            dict[str, object],
            dict[str, bool],
            dict[str, bool],
            dict[str, tuple[str, bool] | None],
            set[str],
        ],
        right: tuple[
            dict[str, str | None],
            dict[str, object],
            dict[str, bool],
            dict[str, bool],
            dict[str, tuple[str, bool] | None],
            set[str],
        ],
    ) -> None:
        (
            left_bindings,
            left_constants,
            left_columns,
            left_attributes,
            left_aliases,
            left_history,
        ) = left
        (
            right_bindings,
            right_constants,
            right_columns,
            right_attributes,
            right_aliases,
            right_history,
        ) = right
        names = (
            set(left_bindings)
            | set(right_bindings)
            | set(left_constants)
            | set(right_constants)
            | set(left_aliases)
            | set(right_aliases)
        )
        history = left_history | right_history
        self.bindings[-1] = {
            name: (
                left_bindings.get(name)
                if self._same_flow_value(
                    left_bindings.get(name),
                    right_bindings.get(name),
                )
                else None
            )
            for name in names
        }
        self.constants[-1] = {
            name: (
                left_constants.get(name)
                if self._same_flow_value(
                    left_constants.get(name),
                    right_constants.get(name),
                )
                else None
            )
            for name in names
        }
        self.column_containers[-1] = {
            name: left_columns.get(name, False) or right_columns.get(name, False)
            for name in names
        }
        self.attribute_containers[-1] = {
            name: left_attributes.get(name, False) or right_attributes.get(name, False)
            for name in names
        }
        self.method_aliases[-1] = {}
        for name in names:
            left_alias = left_aliases.get(name)
            right_alias = right_aliases.get(name)
            if self._same_flow_value(left_alias, right_alias):
                merged_alias = left_alias
            elif name in history or left_alias is not None or right_alias is not None:
                merged_alias = _OPAQUE_METHOD_ALIAS
                history.add(name)
            else:
                merged_alias = None
            self.method_aliases[-1][name] = merged_alias
        self.method_alias_history[-1] = history

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        before = self._flow_state()
        for statement in node.body:
            self.visit(statement)
        body_state = self._flow_state()
        self._restore_flow_state(before)
        for statement in node.orelse:
            self.visit(statement)
        else_state = self._flow_state()
        self._merge_flow_states(body_state, else_state)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        before = self._flow_state()
        self.visit(node.body)
        body_state = self._flow_state()
        self._restore_flow_state(before)
        self.visit(node.orelse)
        else_state = self._flow_state()
        self._merge_flow_states(body_state, else_state)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_access_target(node.target)
        before = self._flow_state()
        values = _static_string_list(node.iter, self.constants)
        if values is None and self._bind_iteration_rows(node.target, node.iter):
            values = ("",)  # rows bound per position; body always analyzed
        else:
            self._bind_iteration_target(
                node.target,
                values,
            )
        for statement in node.body:
            self.visit(statement)
        body_state = self._flow_state()
        if values == ():
            self._restore_flow_state(before)
        elif values is None:
            self._merge_flow_states(before, body_state)
        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        first_generator = node.generators[0]
        self.visit(first_generator.iter)
        first_values = _static_string_list(first_generator.iter, self.constants)
        direct_class_body = bool(
            self.class_lexical_scope_depths
            and len(self.scope_kinds) == self.class_lexical_scope_depths[-1]
        )
        class_body_state = self._flow_state() if direct_class_body else None
        if class_body_state is not None:
            self._restore_flow_state(self.class_lexical_flow_states[-1])
        self.bindings.append({})
        self.constants.append({})
        self.column_containers.append({})
        self.attribute_containers.append({})
        self.method_aliases.append({})
        self.method_alias_history.append(set())
        self.assignment_counts.append({})
        self.scope_kinds.append("comprehension")
        for index, generator in enumerate(node.generators):
            values = first_values
            if index:
                self.visit(generator.iter)
                values = _static_string_list(generator.iter, self.constants)
            self._visit_access_target(generator.target)
            if values is None and self._bind_iteration_rows(
                generator.target, generator.iter
            ):
                pass
            else:
                self._bind_iteration_target(
                    generator.target,
                    values,
                )
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.assignment_counts.pop()
        self.scope_kinds.pop()
        self.method_alias_history.pop()
        self.method_aliases.pop()
        self.attribute_containers.pop()
        self.column_containers.pop()
        self.constants.pop()
        self.bindings.pop()
        if class_body_state is not None:
            self._restore_flow_state(class_body_state)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        """Type parameters are not executable column selectors."""

    def _visit_function_definition_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Visit expressions evaluated when a function is defined."""

        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def _visit_class_definition(
        self,
        node: ast.ClassDef,
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        """Visit a class body without leaking its namespace or annotations."""

        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)

        caller_state = self._flow_state()
        lexical_state = (
            self.class_lexical_flow_states[-1]
            if self.class_lexical_flow_states
            else caller_state
        )
        self._restore_flow_state(lexical_state)
        self.class_lexical_flow_states.append(lexical_state)
        self.class_lexical_scope_depths.append(len(self.scope_kinds))
        deferred: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_function_definition_expressions(statement)
                deferred.append(statement)
            elif isinstance(statement, ast.ClassDef):
                deferred.extend(self._visit_class_definition(statement))
            else:
                self.visit(statement)
        self.class_lexical_scope_depths.pop()
        self.class_lexical_flow_states.pop()
        self._restore_flow_state(caller_state)
        return deferred

    def _visit_scope_statements(self, body: list[ast.stmt]) -> None:
        deferred: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for statement in body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_function_definition_expressions(statement)
                deferred.append(statement)
            elif isinstance(statement, ast.ClassDef):
                deferred.extend(self._visit_class_definition(statement))
            else:
                self.visit(statement)
        for function in deferred:
            self._visit_function(function, visit_definition_expressions=False)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        deferred = self._visit_class_definition(node)
        for function in deferred:
            self._visit_function(function, visit_definition_expressions=False)

    def visit_Module(self, node: ast.Module) -> None:
        counts = _scope_assignment_counts(
            node.body,
            nested_declaration=ast.Global,
        )
        self.assignment_counts[0] = counts
        for name in counts:
            self.bindings[0].setdefault(name, None)
            self.constants[0].setdefault(name, None)
            self.column_containers[0].setdefault(name, False)
            self.attribute_containers[0].setdefault(name, False)
            self.method_aliases[0].setdefault(name, None)
        self._visit_scope_statements(node.body)

    def _unstable_outer_names(self) -> dict[str, bool]:
        unstable: dict[str, bool] = {}
        seen: set[str] = set()
        for index in range(len(self.constants) - 1, -1, -1):
            names = (
                set(self.assignment_counts[index])
                | set(self.constants[index])
                | self.method_alias_history[index]
            )
            for name in names - seen:
                seen.add(name)
                if self.assignment_counts[index].get(name, 0) != 1:
                    unstable[name] = name in self.method_alias_history[index]
        return unstable

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        visit_definition_expressions: bool = True,
    ) -> None:
        if visit_definition_expressions:
            self._visit_function_definition_expressions(node)
        direct_class_body = bool(
            self.class_lexical_scope_depths
            and len(self.scope_kinds) == self.class_lexical_scope_depths[-1]
        )
        class_body_state = self._flow_state() if direct_class_body else None
        if class_body_state is not None:
            self._restore_flow_state(self.class_lexical_flow_states[-1])
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        parameter_names = tuple(argument.arg for argument in arguments)
        if node.args.vararg is not None:
            parameter_names += (node.args.vararg.arg,)
        if node.args.kwarg is not None:
            parameter_names += (node.args.kwarg.arg,)
        counts = _scope_assignment_counts(
            node.body,
            parameters=parameter_names,
        )
        unstable_outer = self._unstable_outer_names()
        local_names = set(counts) | set(unstable_outer)
        local: dict[str, str | None] = dict.fromkeys(local_names)
        containers = {
            argument.arg: argument.annotation is None
            and argument.arg not in {"self", "cls"}
            for argument in arguments
        }
        attribute_containers = {
            argument.arg: (
                argument.annotation is None and argument.arg not in {"self", "cls"}
            )
            or (
                argument.annotation is not None
                and any(
                    marker in ast.unparse(argument.annotation)
                    for marker in ("DataFrame", "Frame")
                )
            )
            for argument in arguments
        }
        if node.args.vararg is not None:
            containers[node.args.vararg.arg] = False
            attribute_containers[node.args.vararg.arg] = False
        if node.args.kwarg is not None:
            containers[node.args.kwarg.arg] = False
            attribute_containers[node.args.kwarg.arg] = False
        for name in local_names:
            containers.setdefault(name, False)
            attribute_containers.setdefault(name, False)
        method_aliases = {
            name: (
                _OPAQUE_METHOD_ALIAS
                if name not in counts and unstable_outer[name]
                else None
            )
            for name in local_names
        }
        alias_history = {
            name
            for name, value in method_aliases.items()
            if value == _OPAQUE_METHOD_ALIAS
        }
        self.bindings.append(local)
        self.constants.append(dict.fromkeys(local_names))
        self.column_containers.append(containers)
        self.attribute_containers.append(attribute_containers)
        self.method_aliases.append(method_aliases)
        self.method_alias_history.append(alias_history)
        self.assignment_counts.append(counts)
        self.scope_kinds.append("function")
        self._visit_scope_statements(node.body)
        self.scope_kinds.pop()
        self.assignment_counts.pop()
        self.method_alias_history.pop()
        self.method_aliases.pop()
        self.attribute_containers.pop()
        self.column_containers.pop()
        self.constants.pop()
        self.bindings.pop()
        if class_body_state is not None:
            self._restore_flow_state(class_body_state)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        direct_class_body = bool(
            self.class_lexical_scope_depths
            and len(self.scope_kinds) == self.class_lexical_scope_depths[-1]
        )
        class_body_state = self._flow_state() if direct_class_body else None
        if class_body_state is not None:
            self._restore_flow_state(self.class_lexical_flow_states[-1])
        arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        parameter_names = tuple(argument.arg for argument in arguments)
        if node.args.vararg is not None:
            parameter_names += (node.args.vararg.arg,)
        if node.args.kwarg is not None:
            parameter_names += (node.args.kwarg.arg,)
        counter = _ScopeAssignmentCounter()
        counter.visit(node.body)
        for name in parameter_names:
            counter._count(name)
        counts = counter.counts
        unstable_outer = self._unstable_outer_names()
        local_names = set(counts) | set(unstable_outer)
        local: dict[str, str | None] = dict.fromkeys(local_names)
        containers = {
            argument.arg: argument.annotation is None
            and argument.arg not in {"self", "cls"}
            for argument in arguments
        }
        attribute_containers = dict(containers)
        if node.args.vararg is not None:
            containers[node.args.vararg.arg] = False
            attribute_containers[node.args.vararg.arg] = False
        if node.args.kwarg is not None:
            containers[node.args.kwarg.arg] = False
            attribute_containers[node.args.kwarg.arg] = False
        for name in local_names:
            containers.setdefault(name, False)
            attribute_containers.setdefault(name, False)
        method_aliases = {
            name: (
                _OPAQUE_METHOD_ALIAS
                if name not in counts and unstable_outer[name]
                else None
            )
            for name in local_names
        }
        alias_history = {
            name
            for name, value in method_aliases.items()
            if value == _OPAQUE_METHOD_ALIAS
        }
        self.bindings.append(local)
        self.constants.append(dict.fromkeys(local_names))
        self.column_containers.append(containers)
        self.attribute_containers.append(attribute_containers)
        self.method_aliases.append(method_aliases)
        self.method_alias_history.append(alias_history)
        self.assignment_counts.append(counts)
        self.scope_kinds.append("lambda")
        self.visit(node.body)
        self.scope_kinds.pop()
        self.assignment_counts.pop()
        self.method_alias_history.pop()
        self.method_aliases.pop()
        self.attribute_containers.pop()
        self.column_containers.pop()
        self.constants.pop()
        self.bindings.pop()
        if class_body_state is not None:
            self._restore_flow_state(class_body_state)

    _MUTATORS = frozenset(
        {
            "append",
            "extend",
            "insert",
            "remove",
            "clear",
            "pop",
            "sort",
            "reverse",
            "update",
            "setdefault",
        }
    )

    def _poison(self, name: str) -> None:
        self.constants[-1][name] = None

    def _visit_getattr(self, node: ast.Call) -> None:
        arguments: list[ast.AST] = []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                if not isinstance(argument.value, (ast.List, ast.Tuple)):
                    self._record(
                        node,
                        "getattr() with hidden or expanded arguments (fail-closed)",
                    )
                    return
                arguments.extend(argument.value.elts)
            else:
                arguments.append(argument)
        if node.keywords or len(arguments) < 2:
            if node.args or node.keywords:
                self._record(
                    node,
                    "getattr() with hidden or expanded arguments (fail-closed)",
                )
            return
        attribute = _static_string_shape(arguments[1], self.constants)
        if attribute is None or _OPAQUE_STRING_PART in attribute:
            if self._attribute_container(arguments[0]):
                self._record(
                    node,
                    "getattr with an unresolvable dynamic attribute (fail-closed)",
                )
        elif _is_source_column_shape(attribute):
            self._record(
                node,
                f"getattr using source column {attribute!r}",
            )

    def _visit_get_call(
        self,
        node: ast.Call,
        *,
        strict_opacity: bool,
    ) -> None:
        key = _call_argument(node, position=0, keyword="key")
        if key is None:
            if strict_opacity and (node.args or node.keywords):
                self._record(
                    node,
                    ".get() with hidden or expanded arguments (fail-closed)",
                )
            return
        resolved = _static_string_values(key, self.constants)
        column = _subscript_source_expression(
            key,
            bindings=self.bindings,
            factory_aliases=self.factory_aliases,
        )
        if resolved is None:
            if column is not None:
                self._record(node, f".get() using {column}")
            elif strict_opacity:
                self._record(
                    node,
                    ".get() with an unresolvable dynamic key (fail-closed)",
                )
            return
        if any(_OPAQUE_STRING_PART in item for item in resolved):
            if strict_opacity or column is not None:
                self._record(
                    node,
                    ".get() with an unresolvable dynamic key (fail-closed)",
                )
            return
        for item in resolved:
            if _is_source_column_shape(item):
                self._record(
                    node,
                    f".get() using source column {item!r}",
                )

    def _visit_query_or_eval_call(
        self,
        node: ast.Call,
        *,
        method: str,
    ) -> None:
        expression = _call_argument(node, position=0, keyword="expr")
        if expression is None and (node.args or node.keywords):
            self._record(
                node,
                f".{method}() with hidden or expanded arguments (fail-closed)",
            )
            return
        if expression is None:
            return
        shape = _static_string_shape(expression, self.constants)
        if shape is None or _OPAQUE_STRING_PART in shape:
            self._record(
                node,
                f".{method}() with an unresolvable dynamic expression (fail-closed)",
            )
            return
        for token in re.findall(r"[\w*?\[\]-]+", shape):
            if _is_source_column_shape(token):
                self._record(
                    node,
                    f".{method}() using source column {token!r}",
                )
                break

    def _visit_filter_call(self, node: ast.Call) -> None:
        items = _call_argument(node, position=0, keyword="items")
        if items is None and (node.args or node.keywords):
            self._record(
                node,
                ".filter() with hidden or expanded arguments (fail-closed)",
            )
            return
        if items is None:
            return
        resolved = _static_string_list(items, self.constants)
        if resolved is None:
            column = _subscript_source_expression(
                items,
                bindings=self.bindings,
                factory_aliases=self.factory_aliases,
            )
            if column is not None:
                self._record(node, f".filter(items=...) using {column}")
            else:
                self._record(
                    node,
                    ".filter(items=...) with an unresolvable dynamic "
                    "list (fail-closed)",
                )
            return
        if any(_OPAQUE_STRING_PART in item for item in resolved):
            self._record(
                node,
                ".filter(items=...) with an unresolvable dynamic list (fail-closed)",
            )
            return
        for item in resolved:
            if _is_source_column_shape(item):
                self._record(
                    node,
                    f".filter(items=...) using source column {item!r}",
                )

    def _visit_strict_method_call(
        self,
        node: ast.Call,
        *,
        method: str,
        strict_opacity: bool,
    ) -> None:
        if method == "getattr":
            self._visit_getattr(node)
        elif method == "get":
            self._visit_get_call(node, strict_opacity=strict_opacity)
        elif method in {"query", "eval"}:
            self._visit_query_or_eval_call(node, method=method)
        else:
            self._visit_filter_call(node)

    def _visit_column_selector(
        self,
        node: ast.AST,
        *,
        receiver: ast.AST,
        selector: ast.AST,
    ) -> None:
        """Apply identical selector checks to [] and direct __getitem__."""

        column_container = self._column_container(receiver)
        resolved = _static_string_values(selector, self.constants)
        column = _subscript_source_expression(
            selector,
            bindings=self.bindings,
            factory_aliases=self.factory_aliases,
        )
        if resolved is None:
            if column is not None:
                self._record(node, f"subscript using {column}")
            elif column_container:
                self._record(
                    node,
                    "subscript with an unresolvable dynamic selector (fail-closed)",
                )
        elif any(_OPAQUE_STRING_PART in item for item in resolved):
            if (
                column_container
                or column is not None
                or any(_is_source_column_shape(item) for item in resolved)
            ):
                self._record(
                    node,
                    "subscript with an unresolvable dynamic selector (fail-closed)",
                )
        else:
            for item in resolved:
                if _is_source_column_shape(item):
                    self._record(
                        node,
                        f"subscript using source column {item!r}",
                    )

    def _visit_getitem_call(self, node: ast.Call) -> None:
        """Treat a direct receiver.__getitem__(key) exactly like receiver[key]."""

        receiver = node.func.value
        if (
            node.keywords
            or len(node.args) != 1
            or isinstance(node.args[0], ast.Starred)
        ):
            if self._column_container(receiver):
                self._record(
                    node,
                    "__getitem__() with hidden or expanded arguments (fail-closed)",
                )
            return
        self._visit_column_selector(
            node,
            receiver=receiver,
            selector=node.args[0],
        )

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.attr in self._MUTATORS
        ):
            self._poison(node.func.value.id)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "__getitem__":
            self._visit_getitem_call(node)
        name = _call_name(node)
        if name in self.factory_aliases:
            self._record(node, f"call to {name}")
        elif name == "getattr":
            self._visit_getattr(node)
        elif (method_alias := self._method_alias(node.func)) is not None:
            if method_alias == _OPAQUE_METHOD_ALIAS:
                self._record(
                    node,
                    "call through an opaque late-bound method alias (fail-closed)",
                )
            else:
                method, strict_opacity = method_alias
                self._visit_strict_method_call(
                    node,
                    method=method,
                    strict_opacity=strict_opacity,
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _OPERATOR_SOURCE_COLUMNS:
            self._record(node, f"attribute {node.attr!r}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        selector = node.slice
        receiver = node.value
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "loc"
            and isinstance(node.slice, ast.Tuple)
            and node.slice.elts
        ):
            selector = node.slice.elts[-1]
            receiver = node.value.value
        self._visit_column_selector(
            node,
            receiver=receiver,
            selector=selector,
        )
        self.generic_visit(node)


def _source_spine_accesses(source: str) -> tuple[str, ...]:
    """Describe strict-surface and static-name contraband violations."""

    tree = ast.parse(source)
    aliases = _factory_aliases(tree)
    visitor = _SourceReadVisitor(
        aliases,
        docstring_value_ids=_docstring_value_ids(tree),
    )
    visitor.visit(tree)
    return tuple(
        f"line {line}:{column + 1}: {description}"
        for line, column, description in sorted(visitor.accesses)
    )


def _non_owner_source_spine_accesses(
    module_name: str,
    source: str,
) -> tuple[str, ...]:
    """Apply the guard unless the module is a reviewed provenance owner."""

    if module_name in _SOURCE_SPINE_PROVENANCE_OWNERS:
        return ()
    return _source_spine_accesses(source)


def _called_function_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (name := _call_name(node)) is not None
    }


def _imported_us_runtime_modules(source: str) -> tuple[str, ...]:
    """Return statically imported, flat ``us_runtime`` module filenames."""

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            module_names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == _US_RUNTIME_IMPORT_PREFIX:
                module_names = (
                    f"{_US_RUNTIME_IMPORT_PREFIX}.{alias.name}"
                    for alias in node.names
                    if (_US_RUNTIME / f"{alias.name}.py").is_file()
                )
            else:
                module_names = (node.module,)
        else:
            continue

        for module_name in module_names:
            prefix = f"{_US_RUNTIME_IMPORT_PREFIX}."
            if not module_name.startswith(prefix):
                continue
            relative = module_name.removeprefix(prefix)
            leaf = relative.split(".", maxsplit=1)[0]
            imported.add(f"{leaf}.py")
    return tuple(sorted(imported))


def _us_runtime_import_graph(
    root: Path,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Resolve the runtime portion of one tool's static import graph."""

    pending = list(_imported_us_runtime_modules(root.read_text()))
    visited: set[str] = set()
    missing: set[str] = set()
    while pending:
        module_name = pending.pop()
        if module_name in visited:
            continue
        visited.add(module_name)
        path = _US_RUNTIME / module_name
        if not path.is_file():
            missing.add(module_name)
            continue
        pending.extend(
            imported
            for imported in _imported_us_runtime_modules(path.read_text())
            if imported not in visited
        )
    return (
        tuple(_US_RUNTIME / name for name in sorted(visited - missing)),
        tuple(sorted(missing)),
    )


def _frame_metadata_drops(source: str) -> tuple[str, ...]:
    """Find Frame rebuilds that carry a mass log but drop stage metadata."""

    drops: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or _call_name(node) != "Frame":
            continue
        keywords = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }
        mass_log = keywords.get("mass_log")
        if not isinstance(mass_log, ast.Attribute) or mass_log.attr != "mass_log":
            continue
        metadata = keywords.get("metadata")
        same_source = (
            isinstance(metadata, ast.Attribute)
            and metadata.attr == "metadata"
            and ast.dump(metadata.value) == ast.dump(mass_log.value)
        )
        if not same_source:
            drops.append(
                f"line {node.lineno}: Frame carrying {ast.unparse(mass_log)} "
                "must carry metadata from the same source frame"
            )
    return tuple(drops)


def _operator_source_channel_reads(source: str) -> tuple[str, ...]:
    """Compatibility name for the all-entity source-identity detector."""

    return _source_spine_accesses(source)


def _unclassified_runtime_modules(module_names: set[str]) -> tuple[str, ...]:
    return tuple(sorted(module_names - _CLASSIFIED_US_RUNTIME_MODULES))


def test_us_runtime_module_classification_is_complete() -> None:
    """Every runtime module is registered or explicitly classified otherwise."""

    actual = {path.name for path in _US_RUNTIME.glob("*.py")}
    overlap = sorted(
        set(_SPINE_BLIND_OPERATOR_MODULES).intersection(_OTHER_US_RUNTIME_MODULES)
    )
    assert not overlap, f"runtime module classifications overlap: {overlap}"
    assert not _unclassified_runtime_modules(actual), (
        "US runtime modules must be classified as registered population "
        "operators or reviewed non-registry modules; unclassified: "
        f"{_unclassified_runtime_modules(actual)}"
    )
    stale = sorted(_CLASSIFIED_US_RUNTIME_MODULES - actual)
    assert not stale, f"runtime module classification contains missing files: {stale}"


def test_runtime_classification_rejects_a_new_unreviewed_module() -> None:
    """Bind fail-closed classification when a future operator file appears."""

    actual = {path.name for path in _US_RUNTIME.glob("*.py")}
    assert _unclassified_runtime_modules(actual | {"future_operator.py"}) == (
        "future_operator.py",
    )


def test_us_runtime_frame_rebuilds_preserve_immutable_metadata() -> None:
    """A transformation carrying mass history must also carry stage receipts."""

    offenders = {
        path.name: drops
        for path in sorted(_US_RUNTIME.glob("*.py"))
        if (drops := _frame_metadata_drops(path.read_text()))
    }
    assert not offenders, (
        "US runtime Frame rebuilds must preserve immutable metadata alongside "
        f"their mass log; found: {offenders}"
    )


def test_runtime_population_operators_are_source_spine_blind() -> None:
    """Every operator obeys the strict-surface and contraband-name contract.

    The guard parses executable syntax rather than searching raw text, so
    comments, annotations, and docstrings may explain the invariant. Executable
    declarations are checked like every other expression. Strict surfaces
    include subscript/``.loc`` reads and writes, attributes/``getattr``,
    canonical factories, and direct or aliased ``get``, ``filter``, ``query``,
    and ``eval`` calls.
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
        accesses = _non_owner_source_spine_accesses(path.name, path.read_text())
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


def test_pool_build_tool_import_graph_is_source_spine_blind() -> None:
    """The wired CLI and every runtime operator it reaches remain blind."""

    missing_tools = [
        str(path) for path in _SPINE_BLIND_BUILD_TOOLS if not path.is_file()
    ]
    assert not missing_tools, (
        f"registered spine-blind build tools are missing: {missing_tools}"
    )

    for tool in _SPINE_BLIND_BUILD_TOOLS:
        runtime_graph, missing_modules = _us_runtime_import_graph(tool)
        assert len(runtime_graph) == 54, (
            f"{tool.name} must reach the pinned 54-module runtime graph; "
            f"reached {len(runtime_graph)}"
        )
        assert not missing_modules, (
            f"{tool.name} imports unresolved US runtime modules: {missing_modules}"
        )
        runtime_names = {path.name for path in runtime_graph}
        missing_required = sorted(_REQUIRED_POOL_RUNTIME_MODULES - runtime_names)
        assert not missing_required, (
            f"{tool.name} does not reach the canonical pool seam modules: "
            f"{missing_required}"
        )
        retired_modules = sorted(_RETIRED_LATE_ASSEMBLY_MODULES & runtime_names)
        assert not retired_modules, (
            f"{tool.name} reaches retired late-assembly modules: {retired_modules}"
        )
        unclassified = _unclassified_runtime_modules(runtime_names)
        assert not unclassified, (
            f"{tool.name} reaches unclassified US runtime modules: {unclassified}"
        )

        offenders: dict[str, tuple[str, ...]] = {}
        for path in (tool, *runtime_graph):
            reads = _non_owner_source_spine_accesses(path.name, path.read_text())
            if reads:
                offenders[str(path.relative_to(_REPOSITORY_ROOT))] = reads
        assert not offenders, (
            "The multispine pool build path must remain source-spine blind. "
            "Only the existing provenance-owner modules may inspect source "
            f"identity; found: {offenders}"
        )


def test_pool_build_import_graph_parser_covers_supported_import_forms() -> None:
    """Direct, aliased, and package-qualified runtime imports are resolved."""

    source = """
import populace.build.us_runtime.acs_transfer as transfer
from populace.build.us_runtime.multispine_pool import run_multispine_pool_path
from populace.build.us_runtime import puf_support
"""
    assert _imported_us_runtime_modules(source) == (
        "acs_transfer.py",
        "multispine_pool.py",
        "puf_support.py",
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


def test_source_spine_ast_guard_detects_reviewer_bypasses() -> None:
    """Pin the detector against every source-identity bypass from review."""

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
    aliased_factory = """
from x import support_channel_column as sc
def op(df):
    return df[sc("person")]
"""
    dynamic_subscript = """
def op(df, entity):
    col = f"{entity}_support_channel"
    return df[col]
"""
    dynamic_getattr = """
def op(row):
    col = "_".join(("person", "support", "channel"))
    return getattr(row, col)
"""
    raw_spine_source_id = """
def op(df):
    return df["person_spine_source_id"]
"""

    assert _source_spine_accesses(direct)
    assert _source_spine_accesses(via_helper)
    assert _source_spine_accesses(clone_index) == ()
    assert _operator_source_channel_reads(named_person_channel)
    assert _source_spine_accesses(aliased_factory)
    assert _source_spine_accesses(dynamic_subscript)
    assert _source_spine_accesses(dynamic_getattr)
    assert _source_spine_accesses(raw_spine_source_id)


def test_named_expression_factory_alias_is_caught() -> None:
    """A walrus-bound canonical factory remains a named factory call."""

    immediate = """
def f():
    return (factory := support_channel_column)("person")
"""
    later = """
def f():
    (factory := support_channel_column)
    return factory("person")
"""

    for source in (immediate, later):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("call to factory" in access for access in accesses)
        assert all("fail-closed" not in access for access in accesses)


def test_source_spine_ast_guard_detects_pandas_string_reads() -> None:
    """Pandas string-based column APIs cannot bypass the structural guard."""

    reviewer_get = """
def op(df):
    return df.get("person_support_channel")
"""
    reviewer_query = """
def op(df):
    return df.query('person_support_channel == "acs"')
"""
    pandas_eval = """
def op(df):
    return df.eval("person_spine_source_id == 1")
"""
    pandas_filter = """
def op(df):
    return df.filter(items=["person_spine"])
"""
    pandas_loc = """
def op(df):
    return df.loc[..., "person_support_channel"]
"""
    benign_reads = """
def op(df):
    df.get("age")
    df.query("age >= 18")
    df.eval("age + 1")
    df.filter(items=["age"])
    return df.loc[..., "age"]
"""

    assert _source_spine_accesses(reviewer_get)
    assert _source_spine_accesses(reviewer_query)
    assert _source_spine_accesses(pandas_eval)
    assert _source_spine_accesses(pandas_filter)
    assert _source_spine_accesses(pandas_loc)
    assert _source_spine_accesses(benign_reads) == ()


def test_typed_parameter_subscript_is_closed_at_static_call_sites() -> None:
    """Typed helper subscripts are permitted; their static producers are not."""

    guarded = """
def select(df: pd.DataFrame, column: str):
    return df[column]
def caller(df: pd.DataFrame):
    return select(df, "person_support_channel")
"""
    benign = guarded.replace('"person_support_channel"', '"age"')
    typed_boundary = """
def select(df: pd.DataFrame, column: str):
    return df[column]
"""

    accesses = _source_spine_accesses(guarded)
    caller_accesses = [access for access in accesses if access.startswith("line 5:")]
    assert caller_accesses
    assert all(
        "contraband source column 'person_support_channel'" in access
        for access in caller_accesses
    )
    assert not any(access.startswith(("line 2:", "line 3:")) for access in accesses)
    assert _source_spine_accesses(benign) == ()
    assert _source_spine_accesses(typed_boundary) == ()


def test_non_strict_subscript_forms_share_the_composition_boundary() -> None:
    """Receiver forms outside local strict inference rely on the producer rule."""

    sources = (
        """
def f(column: str):
    df = pd.DataFrame()
    return df[column]
""",
        """
def f(frame: Frame, column: str):
    return frame.table("person")[column]
""",
        """
def f(df: pd.DataFrame):
    return df[load_runtime_column()]
""",
    )

    for source in sources:
        assert _source_spine_accesses(source) == (), source


def test_function_definition_expressions_use_definition_time_constants() -> None:
    """Defaults and decorators resolve before later enclosing rebindings."""

    guarded_default = """
PREFIX = "person_support"
def f(column=PREFIX + "_channel"):
    return column
PREFIX = "age"
"""
    guarded_decorator = """
PREFIX = "person_support"
@register(PREFIX + "_channel")
def f():
    pass
PREFIX = "age"
"""
    nested_guarded_default = """
def outer():
    prefix = "person_support"
    def inner(column=prefix + "_channel"):
        return column
    prefix = "age"
    return inner
"""
    benign_default = """
PREFIX = "age"
def f(column=PREFIX + "_channel"):
    return column
PREFIX = "person_support"
"""

    for source in (guarded_default, guarded_decorator, nested_guarded_default):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in access for access in accesses)
    assert _source_spine_accesses(benign_default) == ()


def test_class_namespaces_do_not_overwrite_enclosing_constant_state() -> None:
    """Sequential class locals are checked but never leak into outer flow."""

    guarded_outer = """
PREFIX = "person_support"
class Spec:
    PREFIX = "age"
sink(PREFIX + "_channel")
"""
    benign_outer = """
PREFIX = "age"
class Spec:
    PREFIX = "person_support"
sink(PREFIX + "_channel")
"""

    guarded_accesses = _source_spine_accesses(guarded_outer)
    assert any("person_support_channel" in access for access in guarded_accesses)
    assert _source_spine_accesses(benign_outer) == ()


def test_class_created_closures_keep_their_real_nonclass_scopes() -> None:
    """Leaving class scope must preserve nested lambda/comprehension bindings."""

    nested_lambda = """
class Spec:
    make = lambda: (
        (prefix := "person_support"),
        lambda: prefix + "_channel",
    )
"""
    comprehension_lambda = """
class Spec:
    VALUES = ("person_support",)
    funcs = [lambda: prefix + "_channel" for prefix in VALUES]
"""

    for source in (nested_lambda, comprehension_lambda):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in access for access in accesses)


@pytest.mark.parametrize(
    ("position", "source"),
    (
        (
            "call argument",
            """
def f(df):
    return helper(df, "person_support_channel")
""",
        ),
        (
            "list element",
            """
VALUE = ["person_support_channel"]
""",
        ),
        (
            "tuple element",
            """
VALUE = ("person_support_channel",)
""",
        ),
        (
            "set element",
            """
VALUE = {"person_support_channel"}
""",
        ),
        (
            "dict key",
            """
VALUE = {"person_support_channel": "source"}
""",
        ),
        (
            "dict value",
            """
VALUE = {"column": "person_support_channel"}
""",
        ),
        (
            "comparison",
            """
def f(value):
    return value == "person_support_channel"
""",
        ),
        (
            "return value",
            """
def f():
    return "person_support_channel"
""",
        ),
        (
            "assignment",
            """
VALUE = "person_support_channel"
""",
        ),
        (
            "function default",
            """
def f(column="person_support_channel"):
    return column
""",
        ),
        (
            "lambda default",
            """
f = lambda column="person_support_channel": column
""",
        ),
    ),
)
def test_contraband_names_are_rejected_in_every_expression_position(
    position: str,
    source: str,
) -> None:
    """Every executable static spelling is a named, non-opaque violation."""

    accesses = _source_spine_accesses(source)
    assert accesses, position
    assert all("person_support_channel" in access for access in accesses)
    assert all("fail-closed" not in access for access in accesses)


@pytest.mark.parametrize(
    ("composition", "expression"),
    (
        ("concatenation", '"person_support" + "_channel"'),
        ("format", '"{}_support_channel".format("person")'),
        ("percent", '"%s_support_channel" % "person"'),
        ("replace", '"person_x".replace("x", "support_channel")'),
        ("multiplication", '"person_" + ("support" * 1) + "_channel"'),
    ),
)
def test_contraband_names_reuse_static_expression_resolution(
    composition: str,
    expression: str,
) -> None:
    """Composition is resolved outside known pandas access surfaces too."""

    source = f"def f():\n    return sink({expression})\n"
    accesses = _source_spine_accesses(source)
    assert accesses, composition
    assert all("person_support_channel" in access for access in accesses)
    assert all("fail-closed" not in access for access in accesses)


@pytest.mark.parametrize(
    ("case_method", "expression"),
    (
        ("lower", '"PERSON_SUPPORT_CHANNEL".lower()'),
        ("upper", '"pErSoN_sUpPoRt_cHaNnEl".upper().lower()'),
        ("casefold", '"PERSON_SUPPORT_CHANNEL".casefold()'),
        (
            "title",
            '"PERSON SUPPORT CHANNEL".title().replace(" ", "_").lower()',
        ),
        ("capitalize", '"PERSON_SUPPORT_CHANNEL".capitalize().lower()'),
    ),
)
def test_static_case_methods_fold_guarded_names(
    case_method: str,
    expression: str,
) -> None:
    """Every documented zero-argument case method resolves static receivers."""

    source = f"def f():\n    return sink({expression})\n"
    accesses = _source_spine_accesses(source)
    assert accesses, case_method
    assert any("person_support_channel" in access for access in accesses)
    assert all("fail-closed" not in access for access in accesses)


def test_contraband_names_resolve_bound_and_enumerated_fragments() -> None:
    """Bound concatenation and static f-string choices expose exact names."""

    bound = """
PREFIX = "person_support"
def f():
    return sink(PREFIX + "_channel")
"""
    enumerated = """
ENTITIES = ("person", "household")
VALUES = {
    f"{entity}_support_channel"
    for entity in ENTITIES
}
"""
    ordinary_tuple = """
ENTITIES = ("person", "age")
VALUE = f"{ENTITIES}_support_channel"
"""

    bound_accesses = _source_spine_accesses(bound)
    assert any("person_support_channel" in access for access in bound_accesses)
    enumerated_accesses = _source_spine_accesses(enumerated)
    assert any("person_support_channel" in access for access in enumerated_accesses)
    assert any("household_support_channel" in access for access in enumerated_accesses)
    assert _source_spine_accesses(ordinary_tuple) == ()


def test_benign_helpers_and_containers_remain_clean() -> None:
    """Ordinary dynamic helpers and statically benign payloads are permitted."""

    source = """
def select(df: pd.DataFrame, column: str):
    return df[column]
def f(df: pd.DataFrame, column="age"):
    payload = [
        "age",
        ("income",),
        {"tenure"},
        {"column": "wages"},
    ]
    return select(df, column), payload
"""
    assert _source_spine_accesses(source) == ()


def test_annotations_and_true_docstrings_are_not_dataflow() -> None:
    """Every annotation form and each scope's real docstring are exempt."""

    source = '''"""person_support_channel"""
type Alias = Literal["person_support_channel"]
class Spec[
    T: Literal["person_support_channel"],
    U = Literal["person_support_channel"],
]:
    """person_support_channel"""
    field: Literal["person_support_channel"]
def f[V: Literal["person_support_channel"]](
    df: pd.DataFrame,
    column: Literal["person_support_channel"],
) -> Literal["person_support_channel"]:
    """person_support_channel"""
    local: Literal["person_support_channel"]
    return df[column]
'''
    later_expression = """
def f():
    pass
    "person_support_channel"
"""

    assert _source_spine_accesses(source) == ()
    assert _source_spine_accesses(later_expression)


def test_documented_out_of_scope_evasions_are_not_caught() -> None:
    """Pin deliberate obfuscation outside this accidental-awareness tripwire.

    Code review and adversarial merge review, rather than this scanner, control
    code written to deceive. Preserving these known misses keeps that boundary
    visible and prevents claims that the scanner is an adversarial sandbox.
    """

    sources = {
        "reverse slicing": """import pandas as pd
def select(df: pd.DataFrame, column: str):
    return df[column]
def op(df):
    column = "nosrep"[::-1] + "_" + "troppus"[::-1] + "_" + "lennahc"[::-1]
    return select(df, column)
""",
        "format_map": """import pandas as pd
def select(df: pd.DataFrame, column: str):
    return df[column]
def op(df):
    column = "{a}_{b}".format_map({"a": "person", "b": "support_channel"})
    return select(df, column)
""",
        "__doc__ mining": '''import pandas as pd
def marker():
    """person_support_channel"""
def select(df: pd.DataFrame, column: str):
    return df[column]
def op(df):
    return select(df, marker.__doc__)
''',
        "__annotations__ mining": """import pandas as pd
class Marker:
    person_support_channel: str
def select(df: pd.DataFrame, column: str):
    return df[column]
def op(df):
    return select(df, next(iter(Marker.__annotations__)))
""",
        "container-indexed strict alias": """def f(df, expr):
    query = (df.query,)[0]
    return query(expr)
""",
    }

    for evasion, source in sources.items():
        assert _source_spine_accesses(source) == (), evasion


def test_attribute_store_surfaces_are_always_visited() -> None:
    """Assignments and loop targets cannot hide guarded attribute writes."""

    source = """
def f(row, values):
    row.person_support_channel = "asec"
    row.household_spine_source_id: str = "source"
    for row.tax_unit_spine in values:
        pass
    row.spm_unit_spine += 1
"""

    accesses = _source_spine_accesses(source)
    for column in (
        "person_support_channel",
        "household_spine_source_id",
        "tax_unit_spine",
        "spm_unit_spine",
    ):
        assert any(column in access for access in accesses), column


def test_reviewed_provenance_owners_are_unaffected_by_contraband_rule() -> None:
    """The same executable spelling is exempt only in a reviewed owner."""

    source = 'COLUMN = "person_support_channel"\n'
    assert _source_spine_accesses(source)
    assert _non_owner_source_spine_accesses("support_provenance.py", source) == ()
    assert _non_owner_source_spine_accesses("future_operator.py", source)


def test_guard_sees_through_static_indirection_and_fails_closed_on_opacity():
    """Sol #583 round-2 bypasses: one level of static indirection must
    resolve, and anything still opaque is a violation by default."""

    bound_query = """
def f(df):
    expr = 'person_support_channel == "acs"'
    return df.query(expr)
"""
    bound_eval = """
def f(df):
    expr = 'person_support_channel == "acs"'
    return df.eval(expr)
"""
    bound_filter = """
def f(df):
    cols = ["person_support_channel"]
    return df.filter(items=cols)
"""
    fstring_query = """
def f(df):
    col = "person_support_channel"
    return df.query(f"{col} == 'acs'")
"""
    opaque_query = """
def f(df, expr):
    return df.query(expr)
"""
    benign_bound = """
def f(df):
    expr = "age >= 18"
    return df.query(expr)
"""
    assert _source_spine_accesses(bound_query)
    assert _source_spine_accesses(bound_eval)
    assert _source_spine_accesses(bound_filter)
    assert _source_spine_accesses(fstring_query)
    accesses = _source_spine_accesses(opaque_query)
    assert accesses and any("fail-closed" in access for access in accesses)
    assert not _source_spine_accesses(benign_bound)


def test_qualified_factory_aliases_and_pair_loops_are_in_scope():
    """Sol #583 round-6 natural-code gaps: module-qualified factory
    aliases and tuple-unpacking loops over static pair containers are
    ordinary code, so they are enforced, not boundary."""

    qualified_alias = """
import populace.build.us_runtime.support_provenance as provenance


def f():
    factory = provenance.support_channel_column
    return factory("person")
"""
    qualified_named_expr = """
import populace.build.us_runtime.support_provenance as provenance


def f():
    return (factory := provenance.support_channel_column)("person")
"""
    pair_loop = """
PAIRS = (("person", "support_channel"),)


def f():
    for entity, suffix in PAIRS:
        sink(f"{entity}_{suffix}")
"""
    pair_comprehension = """
PAIRS = (("person", "support_channel"),)


def f():
    return [f"{entity}_{suffix}" for entity, suffix in PAIRS]
"""
    benign_pair_loop = """
PAIRS = (("person", "age"),)


def f():
    for entity, suffix in PAIRS:
        sink(f"{entity}_{suffix}")
"""
    for source in (
        qualified_alias,
        qualified_named_expr,
        pair_loop,
        pair_comprehension,
    ):
        assert _source_spine_accesses(source), source
    assert _source_spine_accesses(benign_pair_loop) == ()


def test_guard_treats_wildcards_hidden_args_and_mutations_as_opaque():
    """Sol #583 round-3 evasions: str.format and parameter interpolation
    resolved to a '*' wildcard and read as benign; kwargs-expansion hid
    the expression; mutation staled a resolved list; a shadowing
    parameter fell through to an outer constant. All are opaque now."""

    format_bound = """
def f(df):
    col = "person_support_channel"
    return df.query("{} == 'acs'".format(col))
"""
    param_fstring = """
def f(df, col):
    return df.query(f"{col} == 'acs'")
"""
    kwargs_hidden = """
def f(df, kw):
    return df.query(**kw)
"""
    mutated_filter = """
def f(df):
    cols = ["age"]
    cols.append("person_support_channel")
    return df.filter(items=cols)
"""
    conditional_bound = """
def f(df, flag):
    col = "age" if flag else "person_support_channel"
    return df.query(f"{col} >= 1")
"""
    shadowed_param = """
col = "person_support_channel"


def f(df, col):
    return df.query(f"{col} >= 1")
"""
    for source in (
        format_bound,
        param_fstring,
        kwargs_hidden,
        mutated_filter,
        conditional_bound,
        shadowed_param,
    ):
        accesses = _source_spine_accesses(source)
        assert accesses, source
    # Precision is preserved where resolution succeeds: format_bound's
    # template resolves fully, so it reports the column, not opacity.
    assert any(
        "person_support_channel" in access
        for access in _source_spine_accesses(format_bound)
    )


def test_conditional_and_loop_flow_joins_never_restore_stale_constants() -> None:
    """Branch disagreement and optional loop execution become opaque."""

    conditional_sources = (
        """
def f(df, flag):
    column = "person_support_channel"
    if flag:
        column = "age"
    return df[column]
""",
        """
def f(df, flag):
    if flag:
        column = "person_support_channel"
    else:
        column = "age"
    return df[column]
""",
        """
def f(df, flag):
    query = df.query
    if flag:
        query = print
    return query("person_support_channel == 1")
""",
        """
def f(df, columns):
    column = "age"
    for column in columns:
        pass
    return df[column]
""",
    )
    empty_loop_guarded = """
def f(df):
    column = "person_support_channel"
    for column in ():
        column = "age"
    return df[column]
"""
    empty_loop_benign = """
def f(df):
    column = "age"
    for column in ():
        column = "income"
    return df[column]
"""

    for source in conditional_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in item for item in accesses)
    guarded_accesses = _source_spine_accesses(empty_loop_guarded)
    assert guarded_accesses
    assert any("person_support_channel" in item for item in guarded_accesses)
    assert _source_spine_accesses(empty_loop_benign) == ()


def test_subscript_selectors_resolve_or_fail_closed() -> None:
    """Every round-4 subscript evasion is named or explicitly opaque."""

    walrus = """
def f(df):
    df[(col := "person_support_channel")]
    return df[col]
"""
    nested_call = """
def f(df):
    def source_column():
        return "person_support_channel"
    return df[source_column()]
"""
    dict_indirection = """
COLS = {"unsafe": "person_support_channel"}
def f(df):
    return df[COLS["unsafe"]]
"""
    multiplication = """
def f(df):
    return df["person_support_channel" * 1]
"""
    percent_format = """
def f(df):
    return df["%s_support_channel" % "person"]
"""
    replace_chain = """
def f(df):
    return df[
        "person-SOURCE-channel"
        .replace("-", "_")
        .replace("SOURCE", "support")
    ]
"""
    comprehension_walrus = """
def f(df):
    col = "age"
    [(col := "person_support_channel") for _ in (0,)]
    return df[col]
"""

    walrus_accesses = _source_spine_accesses(walrus)
    walrus_subscripts = [
        access for access in walrus_accesses if "subscript using" in access
    ]
    assert len(walrus_subscripts) == 2
    assert all("person_support_channel" in access for access in walrus_subscripts)
    comprehension_accesses = _source_spine_accesses(comprehension_walrus)
    assert comprehension_accesses
    assert all("person_support_channel" in access for access in comprehension_accesses)

    for source in (multiplication, percent_format, replace_chain):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert all("person_support_channel" in access for access in accesses)
        assert all("fail-closed" not in access for access in accesses)

    for source in (nested_call, dict_indirection):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in access for access in accesses)


def test_dunder_getitem_matches_subscript_selector_checks() -> None:
    """Direct __getitem__ calls have the same static and opacity boundary."""

    equivalent_sources = (
        (
            """
def f(df):
    return df["person_support_channel"]
""",
            """
def f(df):
    return df.__getitem__("person_support_channel")
""",
            "person_support_channel",
        ),
        (
            """
def f(df, column):
    return df[column]
""",
            """
def f(df, column):
    return df.__getitem__(column)
""",
            "fail-closed",
        ),
    )
    benign_sources = (
        """
def f(df):
    return df["age"]
""",
        """
def f(df):
    return df.__getitem__("age")
""",
        """
def f(df: pd.DataFrame, column: str):
    return df[column]
""",
        """
def f(df: pd.DataFrame, column: str):
    return df.__getitem__(column)
""",
    )

    for subscript, dunder, expected in equivalent_sources:
        for source in (subscript, dunder):
            accesses = _source_spine_accesses(source)
            assert accesses, source
            assert any(expected in access for access in accesses)
    for source in benign_sources:
        assert _source_spine_accesses(source) == (), source


def test_static_selector_extensions_are_shared_by_strict_pandas_calls() -> None:
    """Mult, percent formatting, and replace resolve at every strict surface."""

    sources = (
        """
def f(df):
    return df.query("person_support_channel == 1" * 1)
""",
        """
def f(df):
    return df.eval("%s_support_channel == 1" % "person")
""",
        """
def f(df):
    return df.query(
        "person_x == 1"
        .replace("x", "support")
        .replace("support", "support_channel")
    )
""",
        """
def f(df):
    return df.query(
        (expr := "person_support_channel == 1")
    )
""",
    )

    for source in sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert all("person_support_channel" in access for access in accesses)
        assert all("fail-closed" not in access for access in accesses)


def test_loop_targets_propagate_every_static_string_choice() -> None:
    """Loop selectors check every static member and shadow dynamic iterables."""

    benign = """
def f(df):
    for col in ("age", "income"):
        df[col]
"""
    guarded = """
COLUMNS = ("age", "person_support_channel", "household_spine")
def f(df):
    for col in COLUMNS:
        df[col]
"""
    dynamic = """
col = "age"
def f(df, columns):
    for col in columns:
        df[col]
"""

    assert _source_spine_accesses(benign) == ()
    guarded_accesses = _source_spine_accesses(guarded)
    assert any("person_support_channel" in item for item in guarded_accesses)
    assert any("household_spine" in item for item in guarded_accesses)
    dynamic_accesses = _source_spine_accesses(dynamic)
    assert dynamic_accesses
    assert all("fail-closed" in item for item in dynamic_accesses)


def test_loop_targets_accept_every_static_string_iterable_form() -> None:
    """Sets, dict keys, strings, and concatenation bind exact choices."""

    benign_sources = (
        """
def f(df):
    for col in {"age", "income"}:
        df[col]
""",
        """
def f(df):
    for col in {"age": 1, "income": 2}:
        df[col]
""",
        """
def f(df):
    for col in "ab":
        df[col]
""",
        """
def f(df):
    for col in ("age",) + ("income",):
        df[col]
""",
        """
def f(df):
    return [df[col] for col in {"age", "income"}]
""",
    )
    guarded_sources = (
        """
def f(df):
    for col in {"age", "person_support_channel"}:
        df[col]
""",
        """
def f(df):
    for col in {"person_support_channel": 1, "age": 2}:
        df[col]
""",
        """
def f(df):
    return [
        df[col]
        for col in ("age",) + ("person_support_channel",)
    ]
""",
    )

    for source in benign_sources:
        assert _source_spine_accesses(source) == (), source
    for source in guarded_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in item for item in accesses)
        assert all("fail-closed" not in item for item in accesses)


def test_comprehension_targets_propagate_static_and_opaque_choices() -> None:
    """All comprehension forms bind targets before visiting their bodies."""

    benign_sources = (
        """
def f(df):
    return [df[col] for col in ("age", "income")]
""",
        """
def f(df):
    return {df[col] for col in ("age", "income")}
""",
        """
def f(df):
    return {col: df[col] for col in ("age", "income")}
""",
        """
def f(df):
    return tuple(df[col] for col in ("age", "income"))
""",
        """
def f(df):
    return df[[col for col in ("age", "income")]]
""",
    )
    guarded_sources = tuple(
        source.replace(
            '("age", "income")',
            '("age", "person_support_channel")',
        )
        for source in benign_sources
    )
    dynamic_sources = tuple(
        source.replace(
            "def f(df):",
            "def f(df, columns):",
        ).replace(
            '("age", "income")',
            "columns",
        )
        for source in benign_sources
    )

    for source in benign_sources:
        assert _source_spine_accesses(source) == (), source
    for source in guarded_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in item for item in accesses)
    for source in dynamic_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in item for item in accesses)


def test_for_entity_format_repro_is_caught_by_name() -> None:
    """Loop/comprehension format and f-string interpolation expand all choices."""

    sources = {
        "for-entity-format": """
def f():
    for entity in ("person", "household"):
        sink("{}_support_channel".format(entity))
""",
        "for-entity-fstring": """
def f():
    for entity in ("person", "household"):
        sink(f"{entity}_support_channel")
""",
        "comprehension-entity-format": """
def f():
    return [
        "{}_support_channel".format(entity)
        for entity in ("person", "household")
    ]
""",
        "comprehension-entity-fstring": """
def f():
    return [
        f"{entity}_support_channel"
        for entity in ("person", "household")
    ]
""",
    }

    for construction, source in sources.items():
        accesses = _source_spine_accesses(source)
        for column in (
            "person_support_channel",
            "household_support_channel",
        ):
            assert any(column in access for access in accesses), construction
        assert all("fail-closed" not in access for access in accesses)


def test_format_fields_resolve_all_static_forms_by_name() -> None:
    """Automatic, indexed, named, converted, and specified fields are exact."""

    unsafe_sources = (
        """
def f(df):
    col = "person_support_channel"
    return df.query("{} == 1".format(col))
""",
        """
def f(df):
    col = "person_support_channel"
    return df.query("{0} == 1".format(col))
""",
        """
def f(df):
    col = "person_support_channel"
    return df.query("{!s} == 1".format(col))
""",
        """
def f(df):
    col = "person_support_channel"
    return df.query("{!r} == 1".format(col))
""",
        """
def f(df):
    col = "person_support_channel"
    return df.query("{:s} == 1".format(col))
""",
        """
def f(df):
    return df.query(
        "{col} == 1".format(col="person_support_channel")
    )
""",
        """
def f(df):
    return df[
        "{entity}_support_channel".format(entity="person")
    ]
""",
        """
def f(df):
    return df.query(
        "{0:{1}} == 1".format("person_support_channel", "s")
    )
""",
        """
def f(df):
    return df.query(
        "{0[column]} == 1".format(
            {"column": "person_support_channel"}
        )
    )
""",
        """
def f(df):
    return df.query(
        "{} == 1".format(*("person_support_channel",))
    )
""",
        """
def f(df):
    return df.query(
        "{column} == 1".format(
            **{"column": "person_support_channel"}
        )
    )
""",
        """
def f(df):
    return df.query(
        "%(entity)s_support_channel == 1" % {"entity": "person"}
    )
""",
    )

    for source in unsafe_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in item for item in accesses)
        assert all("fail-closed" not in item for item in accesses)


def test_format_fields_preserve_benign_precision_and_opaque_failures() -> None:
    """Fully static benign fields pass; unresolved fields fail closed."""

    benign_sources = (
        """
def f(df):
    return df.query("{col} >= 18".format(col="age"))
""",
        """
def f(df):
    return df.query("{0!s:>3} >= 18".format("age"))
""",
        """
def f(df):
    return df.query("age * 2 >= 18")
""",
        """
def f(df):
    return df.query("{{age}} == {{age}}".format())
""",
        """
def f(df):
    return df.query("{0:{1}} >= 18".format("age", "s"))
""",
        """
def f(df):
    return df.query("{0[column]} >= 18".format({"column": "age"}))
""",
        """
def f(df):
    return df.query("{} >= 18".format(*("age",)))
""",
        """
def f(df):
    return df.query("{column} >= 18".format(**{"column": "age"}))
""",
        """
def f(df):
    return df.query("%(column)s >= 18" % {"column": "age"})
""",
    )
    opaque_sources = (
        """
def f(df, col):
    return df.query("{col} == 1".format(col=col))
""",
        """
def f(df):
    return df.eval("{missing} == 1".format())
""",
        """
def f(df, values):
    return df.query("{} == 1".format(*values))
""",
        """
def f(df, values):
    return df.query("{col} == 1".format(**values))
""",
        """
def f(df, values):
    return df.query("{1} == 1".format(*values, "age"))
""",
    )

    for source in benign_sources:
        assert _source_spine_accesses(source) == (), source
    for source in opaque_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in item for item in accesses)


def test_strict_method_aliases_match_direct_column_checks() -> None:
    """Aliases of query, eval, filter, and get retain exact strict behavior."""

    unsafe = {
        "query": '"person_support_channel == 1"',
        "eval": '"person_support_channel == 1"',
        "filter": 'items=["person_support_channel"]',
        "get": '"person_support_channel"',
    }
    benign = {
        "query": '"age >= 18"',
        "eval": '"age + 1"',
        "filter": 'items=["age", "income"]',
        "get": '"age"',
    }

    for method, arguments in unsafe.items():
        direct = f"""
def f(df):
    return df.{method}({arguments})
"""
        aliased = f"""
def f(df):
    method = df.{method}
    return method({arguments})
"""
        direct_accesses = _source_spine_accesses(direct)
        alias_accesses = _source_spine_accesses(aliased)
        assert direct_accesses, direct
        assert alias_accesses, aliased
        assert any("person_support_channel" in item for item in alias_accesses)
        assert all("fail-closed" not in item for item in alias_accesses)

    for method, arguments in benign.items():
        source = f"""
def f(df):
    method = df.{method}
    return method({arguments})
"""
        assert _source_spine_accesses(source) == (), source


def test_strict_method_aliases_fail_closed_and_shadow_precisely() -> None:
    """Opaque alias arguments/rebindings fail; parameters stop stale aliases."""

    opaque_sources = (
        """
def f(df, expr):
    query = df.query
    return query(expr)
""",
        """
def f(df, expr):
    evaluate = df.eval
    return evaluate(expr)
""",
        """
def f(df, columns):
    select = df.filter
    return select(items=columns)
""",
        """
def f(df, key):
    get = df.get
    return get(key)
""",
        """
def f(df, kwargs):
    query = df.query
    return query(**kwargs)
""",
    )
    rebound_source = """
def f(df):
    query = df.query
    query = print
    return query("age >= 18")
"""
    parameter_shadow = """
query = df.query
def f(query):
    return query("age >= 18")
"""

    for source in opaque_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in item for item in accesses)
    rebound_accesses = _source_spine_accesses(rebound_source)
    assert rebound_accesses
    assert any("fail-closed" in item for item in rebound_accesses)
    assert _source_spine_accesses(parameter_shadow) == ()


def test_strict_method_aliases_cover_expression_and_structural_bindings() -> None:
    """Walrus and unpacked aliases retain strict method identity."""

    guarded_sources = (
        """
def f(df):
    return (q := df.query)("person_support_channel == 1")
""",
        """
def f(df):
    query, printer = (df.query, print)
    return query("person_support_channel == 1")
""",
    )
    benign_sources = (
        """
def f(df):
    return (q := df.query)("age >= 18")
""",
        """
def f(df):
    query, printer = (df.query, print)
    return query("age >= 18")
""",
    )

    for source in guarded_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in item for item in accesses)
    for source in benign_sources:
        assert _source_spine_accesses(source) == (), source

    opaque = """
def f(df, expr):
    return (q := df.query)(expr)
"""
    opaque_accesses = _source_spine_accesses(opaque)
    assert opaque_accesses
    assert any("fail-closed" in access for access in opaque_accesses)


def test_getattr_column_access_resolves_or_fails_closed() -> None:
    """Static attributes are checked by name; dynamic table attrs are opaque."""

    static_alias = """
def f(df):
    query = getattr(df, "query")
    return query("person_support_channel == 1")
"""
    immediate_alias = """
def f(df):
    return getattr(df, "eval")("person_support_channel == 1")
"""
    guarded_attribute = """
def f(df):
    return getattr(df, "person_support_channel")
"""
    dynamic_attribute = """
def f(df, attribute):
    return getattr(df, attribute)
"""
    aliased_getattr = """
def f(df):
    lookup = getattr
    return lookup(df, "person_support_channel")
"""
    starred_getattr = """
def f(df):
    return getattr(*(df, "person_support_channel"))
"""
    opaque_starred_getattr = """
def f(arguments):
    return getattr(*arguments)
"""
    benign_attribute = """
def f(df):
    attribute = "age"
    return getattr(df, attribute)
"""
    generic_object = """
def f(obj: object, attribute: str):
    return getattr(obj, attribute)
"""

    for source in (
        static_alias,
        immediate_alias,
        guarded_attribute,
        aliased_getattr,
        starred_getattr,
    ):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in item for item in accesses)
    dynamic_accesses = _source_spine_accesses(dynamic_attribute)
    assert dynamic_accesses
    assert any("fail-closed" in item for item in dynamic_accesses)
    opaque_starred_accesses = _source_spine_accesses(opaque_starred_getattr)
    assert opaque_starred_accesses
    assert any("fail-closed" in item for item in opaque_starred_accesses)
    assert _source_spine_accesses(benign_attribute) == ()
    assert _source_spine_accesses(generic_object) == ()


def test_closure_free_names_obey_late_binding_assignment_counts() -> None:
    """Multi-assignment free names are opaque; stable names remain exact."""

    late_bound = """
def outer(df):
    expr = "age >= 18"
    def inner():
        return df.query(expr)
    expr = "person_support_channel == 1"
    return inner()
"""
    module_late_bound = """
expr = "age >= 18"
def f(df):
    return df.query(expr)
expr = "person_support_channel == 1"
"""
    stable_guarded = """
def outer(df):
    expr = "person_support_channel == 1"
    def inner():
        return df.query(expr)
    return inner()
"""
    stable_benign = """
def outer(df):
    expr = "age >= 18"
    def inner():
        return df.query(expr)
    return inner()
"""
    later_local_shadow = """
expr = "income >= 0"
def outer(df):
    def inner():
        return df.query(expr)
    expr = "age >= 18"
    return inner()
"""
    lambda_late_bound = """
def outer(df):
    expr = "age >= 18"
    inner = lambda: df.query(expr)
    expr = "person_support_channel == 1"
    return inner()
"""
    comprehension_rebound = """
def outer(df):
    expr = "age >= 18"
    def inner():
        return df.query(expr)
    [(expr := "person_support_channel == 1") for _ in (0,)]
    return inner()
"""
    nonlocal_rebound = """
def outer(df):
    expr = "age >= 18"
    def rebind():
        nonlocal expr
        expr = "person_support_channel == 1"
    def inner():
        return df.query(expr)
    rebind()
    return inner()
"""
    global_rebound = """
expr = "age >= 18"
def rebind():
    global expr
    expr = "person_support_channel == 1"
def f(df):
    rebind()
    return df.query(expr)
"""

    for source in (
        late_bound,
        module_late_bound,
        lambda_late_bound,
        comprehension_rebound,
        nonlocal_rebound,
        global_rebound,
    ):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in item for item in accesses)
    guarded_accesses = _source_spine_accesses(stable_guarded)
    assert guarded_accesses
    assert any("person_support_channel" in item for item in guarded_accesses)
    assert all("fail-closed" not in item for item in guarded_accesses)
    assert _source_spine_accesses(stable_benign) == ()
    assert _source_spine_accesses(later_local_shadow) == ()


def test_closure_method_aliases_are_stable_or_explicitly_opaque() -> None:
    """Late alias rebinding cannot turn a strict call into a silent Name call."""

    stable_alias = """
def outer(df):
    query = df.query
    def inner():
        return query("person_support_channel == 1")
    return inner()
"""
    rebound_alias = """
def outer(df):
    query = df.query
    def inner():
        return query("person_support_channel == 1")
    query = print
    return inner()
"""

    stable_accesses = _source_spine_accesses(stable_alias)
    assert stable_accesses
    assert any("person_support_channel" in item for item in stable_accesses)
    rebound_accesses = _source_spine_accesses(rebound_alias)
    assert rebound_accesses
    assert any("fail-closed" in item for item in rebound_accesses)


@pytest.mark.parametrize(
    ("evasion", "source"),
    (
        (
            "round2-bound-query",
            """
def f(df):
    expr = "person_support_channel == 1"
    return df.query(expr)
""",
        ),
        (
            "round2-bound-eval",
            """
def f(df):
    expr = "person_support_channel == 1"
    return df.eval(expr)
""",
        ),
        (
            "round2-bound-filter",
            """
def f(df):
    columns = ["person_support_channel"]
    return df.filter(items=columns)
""",
        ),
        (
            "round2-bound-fstring",
            """
def f(df):
    column = "person_support_channel"
    return df.query(f"{column} == 1")
""",
        ),
        (
            "round3-parameter-fstring",
            """
def f(df, column):
    return df.query(f"{column} == 1")
""",
        ),
        (
            "round3-expanded-kwargs",
            """
def f(df, kwargs):
    return df.query(**kwargs)
""",
        ),
        (
            "round3-shadowed-parameter",
            """
column = "person_support_channel"
def f(df, column):
    return df.query(f"{column} == 1")
""",
        ),
        (
            "round3-conditional",
            """
def f(df, flag):
    column = "age" if flag else "person_support_channel"
    return df.query(f"{column} == 1")
""",
        ),
        (
            "round3-mutated-list",
            """
def f(df):
    columns = ["age"]
    columns.append("person_support_channel")
    return df.filter(items=columns)
""",
        ),
        (
            "format-automatic",
            """
def f(df):
    return df.query("{} == 1".format("person_support_channel"))
""",
        ),
        (
            "format-indexed",
            """
def f(df):
    return df.query("{0} == 1".format("person_support_channel"))
""",
        ),
        (
            "format-named",
            """
def f(df):
    return df.query(
        "{column} == 1".format(column="person_support_channel")
    )
""",
        ),
        (
            "format-convert-s",
            """
def f(df):
    return df.query("{!s} == 1".format("person_support_channel"))
""",
        ),
        (
            "format-convert-r",
            """
def f(df):
    return df.query("{!r} == 1".format("person_support_channel"))
""",
        ),
        (
            "format-spec",
            """
def f(df):
    return df.query("{:s} == 1".format("person_support_channel"))
""",
        ),
        (
            "subscript-walrus",
            """
def f(df):
    return df[(column := "person_support_channel")]
""",
        ),
        (
            "subscript-nested-call",
            """
def f(df):
    def column():
        return "person_support_channel"
    return df[column()]
""",
        ),
        (
            "subscript-dict-indirection",
            """
COLUMNS = {"source": "person_support_channel"}
def f(df):
    return df[COLUMNS["source"]]
""",
        ),
        (
            "subscript-multiplication",
            """
def f(df):
    return df["person_support_channel" * 1]
""",
        ),
        (
            "subscript-percent-format",
            """
def f(df):
    return df["%s_support_channel" % "person"]
""",
        ),
        (
            "subscript-replace-chain",
            """
def f(df):
    return df["person_x".replace("x", "support_channel")]
""",
        ),
        (
            "aliased-query",
            """
def f(df):
    query = df.query
    return query("person_support_channel == 1")
""",
        ),
        (
            "aliased-eval",
            """
def f(df):
    evaluate = df.eval
    return evaluate("person_support_channel == 1")
""",
        ),
        (
            "aliased-filter",
            """
def f(df):
    select = df.filter
    return select(items=["person_support_channel"])
""",
        ),
        (
            "aliased-get",
            """
def f(df):
    get = df.get
    return get("person_support_channel")
""",
        ),
        (
            "dynamic-getattr",
            """
def f(df, attribute):
    return getattr(df, attribute)
""",
        ),
        (
            "late-bound-closure",
            """
def outer(df):
    expr = "age >= 18"
    def inner():
        return df.query(expr)
    expr = "person_support_channel == 1"
    return inner()
""",
        ),
        (
            "round5-static-lower",
            """
import pandas as pd
def select(df: pd.DataFrame, column: str):
    return df[column]
def op(df):
    column = "PERSON_SUPPORT_CHANNEL".lower()
    return select(df, column)
""",
        ),
        (
            "round5-for-entity-format",
            """
def f():
    for entity in ("person", "household"):
        sink("{}_support_channel".format(entity))
""",
        ),
        (
            "round5-dunder-getitem",
            """
def f(df, column):
    return df.__getitem__(column)
""",
        ),
        (
            "round5-factory-walrus",
            """
def f():
    return (factory := support_channel_column)("person")
""",
        ),
        (
            "round5-query-walrus",
            """
def f(df, expr):
    return (q := df.query)(expr)
""",
        ),
    ),
)
def test_every_review_evasion_is_caught(evasion: str, source: str) -> None:
    """Enumerated in-scope rounds 2-5 cases resolve or fail closed."""

    assert _source_spine_accesses(source), evasion


@pytest.mark.parametrize(
    ("control", "source"),
    (
        (
            "literal-query",
            """
def f(df):
    return df.query("age >= 18")
""",
        ),
        (
            "bound-query",
            """
def f(df):
    expr = "age >= 18"
    return df.query(expr)
""",
        ),
        (
            "resolved-fstring",
            """
def f(df):
    column = "age"
    return df.query(f"{column} >= 18")
""",
        ),
        (
            "bare-format",
            """
def f(df):
    column = "age"
    return df.query("{} >= 18".format(column))
""",
        ),
        (
            "named-format",
            """
def f(df):
    return df.query("{column} >= 18".format(column="age"))
""",
        ),
        (
            "bound-filter-list",
            """
def f(df):
    columns = ["age", "income"]
    return df.filter(items=columns)
""",
        ),
        (
            "literal-subscript",
            """
def f(df):
    return df["age"]
""",
        ),
        (
            "bound-subscript",
            """
def f(df):
    column = "age"
    return df[column]
""",
        ),
        (
            "static-loop",
            """
def f(df):
    for column in ("age", "income"):
        df[column]
""",
        ),
        (
            "static-comprehension",
            """
def f(df):
    return [df[column] for column in ("age", "income")]
""",
        ),
        (
            "aliased-query",
            """
def f(df):
    query = df.query
    return query("age >= 18")
""",
        ),
        (
            "aliased-eval",
            """
def f(df):
    evaluate = df.eval
    return evaluate("age + 1")
""",
        ),
        (
            "aliased-filter",
            """
def f(df):
    select = df.filter
    return select(items=["age", "income"])
""",
        ),
        (
            "aliased-get",
            """
def f(df):
    get = df.get
    return get("age")
""",
        ),
        (
            "static-getattr",
            """
def f(df):
    return getattr(df, "age")
""",
        ),
        (
            "single-assignment-closure",
            """
def outer(df):
    expr = "age >= 18"
    def inner():
        return df.query(expr)
    return inner()
""",
        ),
        (
            "literal-expression-multiplication",
            """
def f(df):
    return df.query("age * 2 >= 18")
""",
        ),
        (
            "typed-row-mask",
            """
def f(values: list[float], mask: list[bool]):
    return values[mask]
""",
        ),
        (
            "typed-mapping-get",
            """
def f(values: dict[str, float], key: str):
    return values.get(key)
""",
        ),
    ),
)
def test_benign_column_access_battery_is_clean(control: str, source: str) -> None:
    """Static benign columns and proven non-column access produce no finding."""

    assert _source_spine_accesses(source) == (), control


def test_source_spine_ast_guard_covers_every_entity_grain() -> None:
    """Every US and benefit-unit grain's source identity is prohibited."""

    for entity in _US_ENTITIES:
        for suffix in ("spine", "spine_source_id", "support_channel"):
            column = f"{entity}_{suffix}"
            source = f'def op(df):\n    return df["{column}"]\n'
            assert _source_spine_accesses(source), column
