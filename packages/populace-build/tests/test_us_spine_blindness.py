"""Structural guard for source-spine-blind US population operators."""

from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path
from string import Formatter

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
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
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
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Name) or value.id not in aliases:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
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
    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        (float, bool, bytes, type(None)),
    ):
        return node.value
    return _OPAQUE_STATIC_VALUE


def _resolve_static_format(
    template: str,
    node: ast.Call,
    constants: list[dict[str, object]],
) -> str:
    """Substitute every statically known ``str.format`` field."""

    formatter = Formatter()
    positional = tuple(
        _OPAQUE_STATIC_VALUE
        if isinstance(argument, ast.Starred)
        else _static_format_value(argument, constants)
        for argument in node.args
    )
    keywords = {
        keyword.arg: _static_format_value(keyword.value, constants)
        for keyword in node.keywords
        if keyword.arg is not None
    }
    expanded_keywords = any(keyword.arg is None for keyword in node.keywords)
    auto_index = 0
    pieces: list[str] = []
    try:
        parsed = tuple(formatter.parse(template))
    except ValueError:
        return _OPAQUE_STRING_PART
    for literal, field_name, format_spec, conversion in parsed:
        pieces.append(literal)
        if field_name is None:
            continue
        lookup_name = field_name
        if field_name == "":
            lookup_name = str(auto_index)
            auto_index += 1
        try:
            value, _ = formatter.get_field(
                lookup_name,
                positional,
                keywords,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            value = _OPAQUE_STATIC_VALUE
        if value is _OPAQUE_STATIC_VALUE or (
            expanded_keywords and not lookup_name.isdecimal()
        ):
            pieces.append(_OPAQUE_STRING_PART)
            continue
        if "{" in format_spec or "}" in format_spec:
            pieces.append(_OPAQUE_STRING_PART)
            continue
        try:
            converted = formatter.convert_field(value, conversion)
            pieces.append(formatter.format_field(converted, format_spec))
        except (TypeError, ValueError):
            pieces.append(_OPAQUE_STRING_PART)
    return "".join(pieces)


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
                if isinstance(value, (str, int, float, bytes, tuple)):
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
    return _OPAQUE_STATIC_VALUE


def _static_string_list(
    node: ast.AST, constants: list[dict[str, object]]
) -> tuple[str, ...] | None:
    """Resolve a static list/tuple of strings, through one Name binding."""

    if isinstance(node, ast.Name):
        for scope in reversed(constants):
            if node.id in scope:
                value = scope[node.id]
                if isinstance(value, tuple) and all(
                    isinstance(item, str) for item in value
                ):
                    return value
                return None
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        items: list[str] = []
        for element in node.elts:
            values = _static_string_values(element, constants)
            if values is None:
                return None
            items.extend(values)
        return tuple(items)
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
                local[name] = values
        return _static_string_values(node.elt, nested_constants)
    return None


def _static_string_values(
    node: ast.AST,
    constants: list[dict[str, object]],
) -> tuple[str, ...] | None:
    shape = _static_string_shape(node, constants)
    if shape is not None:
        return (shape,)
    return _static_string_list(node, constants)


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

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.visit(node.generators[0].iter)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.visit(node.generators[0].iter)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.visit(node.generators[0].iter)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.visit(node.generators[0].iter)


def _scope_assignment_counts(
    body: list[ast.stmt],
    *,
    parameters: tuple[str, ...] = (),
) -> dict[str, int]:
    counter = _ScopeAssignmentCounter()
    for statement in body:
        counter.visit(statement)
    for name in parameters:
        counter._count(name)
    return counter.counts


class _SourceReadVisitor(ast.NodeVisitor):
    def __init__(self, factory_aliases: set[str]) -> None:
        self.factory_aliases = factory_aliases
        self.bindings: list[dict[str, str | None]] = [{}]
        self.constants: list[dict[str, object]] = [{}]
        self.column_containers: list[dict[str, bool]] = [{}]
        self.attribute_containers: list[dict[str, bool]] = [{}]
        self.method_aliases: list[dict[str, tuple[str, bool] | None]] = [{}]
        self.method_alias_history: list[set[str]] = [set()]
        self.assignment_counts: list[dict[str, int]] = [{}]
        self.accesses: set[tuple[int, int, str]] = set()

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
        if isinstance(node, ast.Name):
            for scope in reversed(self.method_aliases):
                if node.id in scope:
                    return scope[node.id]
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

    def _bind(self, targets: list[ast.AST], value: ast.AST) -> None:
        description = self._expression(value)
        constant: object = _static_string_shape(value, self.constants)
        if constant is None:
            constant = _static_string_list(value, self.constants)
        if constant is None:
            constant = _static_integer(value, self.constants)
        container = self._column_container(value)
        attribute_container = self._attribute_container(value)
        method_alias = self._method_alias(value)
        for target in targets:
            for name in _assigned_names(target):
                self.bindings[-1][name] = description
                # None is an explicit opaque shadow: lookups must stop here,
                # never fall through to a stale outer binding (shadowed
                # parameters and conditional reassignments — sol round 3).
                self.constants[-1][name] = constant
                self.column_containers[-1][name] = container
                self.attribute_containers[-1][name] = attribute_container
                self.method_aliases[-1][name] = method_alias
                if method_alias is not None:
                    self.method_alias_history[-1].add(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                self.visit(target)
                if isinstance(target.value, ast.Name):
                    self._poison(target.value.id)
        self._bind(list(node.targets), node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.visit(node.target)
        for name in _assigned_names(node.target):
            self._poison(name)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            for name in _assigned_names(node.target):
                self.bindings[-1][name] = None
                self.constants[-1][name] = None
                self.column_containers[-1][name] = False
                self.attribute_containers[-1][name] = False
                self.method_aliases[-1][name] = None
            return
        self.visit(node.value)
        if isinstance(node.target, ast.Subscript):
            self.visit(node.target)
        self._bind([node.target], node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind([node.target], node.value)

    def _bind_iteration_target(
        self,
        target: ast.AST,
        values: tuple[str, ...] | None,
    ) -> None:
        for name in _assigned_names(target):
            self.bindings[-1][name] = None
            self.constants[-1][name] = values
            self.column_containers[-1][name] = False
            self.attribute_containers[-1][name] = False
            self.method_aliases[-1][name] = None

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_iteration_target(
            node.target,
            _static_string_list(node.iter, self.constants),
        )
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        self.bindings.append({})
        self.constants.append({})
        self.column_containers.append({})
        self.attribute_containers.append({})
        self.method_aliases.append({})
        self.method_alias_history.append(set())
        self.assignment_counts.append({})
        for generator in node.generators:
            self.visit(generator.iter)
            self._bind_iteration_target(
                generator.target,
                _static_string_list(generator.iter, self.constants),
            )
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.assignment_counts.pop()
        self.method_alias_history.pop()
        self.method_aliases.pop()
        self.attribute_containers.pop()
        self.column_containers.pop()
        self.constants.pop()
        self.bindings.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def _visit_scope_statements(self, body: list[ast.stmt]) -> None:
        deferred: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for statement in body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                deferred.append(statement)
            else:
                self.visit(statement)
        for function in deferred:
            self.visit(function)

    def visit_Module(self, node: ast.Module) -> None:
        counts = _scope_assignment_counts(node.body)
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
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
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
        self._visit_scope_statements(node.body)
        self.assignment_counts.pop()
        self.method_alias_history.pop()
        self.method_aliases.pop()
        self.attribute_containers.pop()
        self.column_containers.pop()
        self.constants.pop()
        self.bindings.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
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
        self.visit(node.body)
        self.assignment_counts.pop()
        self.method_alias_history.pop()
        self.method_aliases.pop()
        self.attribute_containers.pop()
        self.column_containers.pop()
        self.constants.pop()
        self.bindings.pop()

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
        if len(node.args) < 2:
            return
        attribute = _static_string_shape(node.args[1], self.constants)
        if attribute is None or _OPAQUE_STRING_PART in attribute:
            if self._attribute_container(node.args[0]):
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
        if method == "get":
            self._visit_get_call(node, strict_opacity=strict_opacity)
        elif method in {"query", "eval"}:
            self._visit_query_or_eval_call(node, method=method)
        else:
            self._visit_filter_call(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.attr in self._MUTATORS
        ):
            self._poison(node.func.value.id)
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
        column_container = self._column_container(node.value)
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "loc"
            and isinstance(node.slice, ast.Tuple)
            and node.slice.elts
        ):
            selector = node.slice.elts[-1]
            column_container = self._column_container(node.value.value)

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
        self.generic_visit(node)


def _source_spine_accesses(source: str) -> tuple[str, ...]:
    """Describe data reads that resolve any entity's source-spine identity."""

    tree = ast.parse(source)
    aliases = _factory_aliases(tree)
    visitor = _SourceReadVisitor(aliases)
    visitor.visit(tree)
    return tuple(
        f"line {line}:{column + 1}: {description}"
        for line, column, description in sorted(visitor.accesses)
    )


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
    """Only provenance owners may resolve any entity's source-spine identity.

    The guard parses executable syntax rather than searching raw text, so
    comments, docstrings, and source-manifest declarations may explain the
    invariant without creating an exception. Data access through a concrete
    column, a dynamic string, ``getattr``, or a canonical factory fails.
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
            if path.name in _SOURCE_SPINE_PROVENANCE_OWNERS:
                continue
            reads = _source_spine_accesses(path.read_text())
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

    walrus_accesses = _source_spine_accesses(walrus)
    assert len(walrus_accesses) == 2
    assert all("person_support_channel" in access for access in walrus_accesses)

    for source in (multiplication, percent_format, replace_chain):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert all("person_support_channel" in access for access in accesses)
        assert all("fail-closed" not in access for access in accesses)

    for source in (nested_call, dict_indirection):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in access for access in accesses)


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
    assert len(guarded_accesses) == 2
    assert any("person_support_channel" in item for item in guarded_accesses)
    assert any("household_spine" in item for item in guarded_accesses)
    dynamic_accesses = _source_spine_accesses(dynamic)
    assert dynamic_accesses
    assert all("fail-closed" in item for item in dynamic_accesses)


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
    """Opaque alias arguments fail; rebinding and parameters stop stale aliases."""

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
    shadowed_sources = (
        """
def f(df):
    query = df.query
    query = print
    return query("age >= 18")
""",
        """
query = df.query
def f(query):
    return query("age >= 18")
""",
    )

    for source in opaque_sources:
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("fail-closed" in item for item in accesses)
    for source in shadowed_sources:
        assert _source_spine_accesses(source) == (), source


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
    benign_attribute = """
def f(df):
    attribute = "age"
    return getattr(df, attribute)
"""
    generic_object = """
def f(obj: object, attribute: str):
    return getattr(obj, attribute)
"""

    for source in (static_alias, immediate_alias, guarded_attribute):
        accesses = _source_spine_accesses(source)
        assert accesses, source
        assert any("person_support_channel" in item for item in accesses)
    dynamic_accesses = _source_spine_accesses(dynamic_attribute)
    assert dynamic_accesses
    assert any("fail-closed" in item for item in dynamic_accesses)
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
expr = "person_support_channel == 1"
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

    for source in (late_bound, module_late_bound, lambda_late_bound):
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


def test_source_spine_ast_guard_covers_every_entity_grain() -> None:
    """Every US and benefit-unit grain's source identity is prohibited."""

    for entity in _US_ENTITIES:
        for suffix in ("spine", "spine_source_id", "support_channel"):
            column = f"{entity}_{suffix}"
            source = f'def op(df):\n    return df["{column}"]\n'
            assert _source_spine_accesses(source), column
