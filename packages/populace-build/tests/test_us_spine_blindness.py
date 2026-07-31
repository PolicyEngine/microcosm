"""Structural guard for source-spine-blind US population operators."""

from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path

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
        (
            candidate.value
            for candidate in node.keywords
            if candidate.arg == keyword
        ),
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


class _SourceReadVisitor(ast.NodeVisitor):
    def __init__(self, factory_aliases: set[str]) -> None:
        self.factory_aliases = factory_aliases
        self.bindings: list[dict[str, str | None]] = [{}]
        self.accesses: set[tuple[int, int, str]] = set()

    def _expression(self, node: ast.AST) -> str | None:
        return _source_expression(
            node,
            bindings=self.bindings,
            factory_aliases=self.factory_aliases,
        )

    def _record(self, node: ast.AST, description: str) -> None:
        self.accesses.add((node.lineno, node.col_offset, description))

    def _bind(self, targets: list[ast.AST], value: ast.AST) -> None:
        description = self._expression(value)
        for target in targets:
            for name in _assigned_names(target):
                self.bindings[-1][name] = description

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        self._bind(list(node.targets), node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            for name in _assigned_names(node.target):
                self.bindings[-1][name] = None
            return
        self.visit(node.value)
        self._bind([node.target], node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind([node.target], node.value)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        local: dict[str, str | None] = {
            argument.arg: None
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        if node.args.vararg is not None:
            local[node.args.vararg.arg] = None
        if node.args.kwarg is not None:
            local[node.args.kwarg.arg] = None
        self.bindings.append(local)
        for statement in node.body:
            self.visit(statement)
        self.bindings.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        local = {
            argument.arg: None
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        self.bindings.append(local)
        self.visit(node.body)
        self.bindings.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name in self.factory_aliases:
            self._record(node, f"call to {name}")
        elif name == "getattr" and len(node.args) >= 2:
            attribute = self._expression(node.args[1])
            if attribute is not None:
                self._record(node, f"getattr using {attribute}")
        elif isinstance(node.func, ast.Attribute) and name == "get":
            key = _call_argument(node, position=0, keyword="key")
            if key is not None:
                column = _subscript_source_expression(
                    key,
                    bindings=self.bindings,
                    factory_aliases=self.factory_aliases,
                )
                if column is not None:
                    self._record(node, f".get() using {column}")
        elif isinstance(node.func, ast.Attribute) and name in {"query", "eval"}:
            expression = _call_argument(node, position=0, keyword="expr")
            if expression is not None:
                column = _pandas_expression_source(expression)
                if column is not None:
                    self._record(node, f".{name}() using {column}")
        elif isinstance(node.func, ast.Attribute) and name == "filter":
            items = _call_argument(node, position=0, keyword="items")
            if items is not None:
                column = _subscript_source_expression(
                    items,
                    bindings=self.bindings,
                    factory_aliases=self.factory_aliases,
                )
                if column is not None:
                    self._record(node, f".filter(items=...) using {column}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _OPERATOR_SOURCE_COLUMNS:
            self._record(node, f"attribute {node.attr!r}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        column = _subscript_source_expression(
            node.slice,
            bindings=self.bindings,
            factory_aliases=self.factory_aliases,
        )
        if column is not None:
            self._record(node, f"subscript using {column}")
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


def test_source_spine_ast_guard_covers_every_entity_grain() -> None:
    """Every US and benefit-unit grain's source identity is prohibited."""

    for entity in _US_ENTITIES:
        for suffix in ("spine", "spine_source_id", "support_channel"):
            column = f"{entity}_{suffix}"
            source = f'def op(df):\n    return df["{column}"]\n'
            assert _source_spine_accesses(source), column
