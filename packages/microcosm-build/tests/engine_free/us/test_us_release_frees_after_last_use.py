"""The US release tool's ``_main`` frees large objects after their last use.

``_main`` is one long function, so a local it never deletes stays alive until
the run ends. These tests read the function's syntax tree, and execute slices
of its statements, to check that each large object is deleted, rebound to
``None``, or never bound to a local, once nothing reads it.
"""

# ruff: noqa: F403, F405
import contextlib
import re
import textwrap
import weakref

from microcosm.build.gate_battery import GateStatus
from microcosm.build.us_runtime.gate_shadow import current_gate_shadow
from microcosm.calibrate import DroppedConstraintMatrix
from test_support.microcosm_build.us_gate_shadow import arm_us_tool_gate_shadows
from test_support.microcosm_build.us_post_export_scoring import *

#: The nine donor-table loaders ``_main`` calls, one per imputation stage.
_DONOR_LOADERS = (
    "load_org_2024_donor",
    "load_scf_2022_auto_loan_donor",
    "load_scf_2022_financial_asset_donor",
    "load_sipp_2023_financial_asset_donor",
    "load_sipp_2023_head_start_donor",
    "load_sipp_2023_ssi_disability_donor",
    "load_sipp_2023_tip_donor",
    "load_sipp_2023_vehicle_donor",
    "load_sipp_2023_voluntary_filing_donor",
)


#: Locals of ``_main`` that hold an object this module checks is freed.
_FREED_NAMES = frozenset(
    {
        "acs_predictor_join_result",
        "base_frame",
        "ladder_outcome",
        "release_engine",
        "result",
    }
)
#: Attributes through which such a local reaches another large object.
_HOLDING_ATTRIBUTES = frozenset(
    {"frame", "matrix", "problem", "refit", "result", "selection"}
)
#: Calls whose result holds their arguments, or a copy of them.
_CONTAINER_CALLS = frozenset(
    {
        "SimpleNamespace",
        "copy",
        "deepcopy",
        "dict",
        "frozenset",
        "list",
        "partial",
        "set",
        "tuple",
    }
)
#: Methods that store their arguments in the object they are called on.
_STORING_METHODS = frozenset(
    {"add", "append", "extend", "insert", "setdefault", "update"}
)
#: The places in ``_main`` that bind another name, or put a container, a copy
#: or a loop variable, on one of those objects or a large part of one, in
#: source order. Any other would keep the object alive after ``_main`` frees
#: it.
_EXPECTED_ALIAS_BINDINGS = (
    # The pre-selection frame on the frozen-support path. It is another object
    # than the base_frame released after the export check, and this change
    # leaves it alive until _main returns (a follow-up).
    "selection_candidate_frame = base_frame",
    # The join's frame becomes the working frame; the result is deleted.
    "base_frame = acs_predictor_join_result.frame",
    # Rebound to the matrix-free result before the export.
    "result = ladder_outcome.result",
    # The outcome's copy holding the matrix-free result replaces the original.
    "ladder_outcome = dataclasses.replace(ladder_outcome, result=result)",
)


class _LargeObject:
    """A stand-in for a frame or table that a weak reference can follow."""


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _arguments(call: ast.Call) -> list[ast.expr]:
    return [*call.args, *(keyword.value for keyword in call.keywords)]


def _holds_a_freed_object(node: ast.AST) -> bool:
    """Whether an expression evaluates to, or contains, a checked object."""
    if isinstance(node, ast.Name):
        return node.id in _FREED_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _HOLDING_ATTRIBUTES and _holds_a_freed_object(node.value)
    if isinstance(node, (ast.Starred, ast.NamedExpr)):
        return _holds_a_freed_object(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_holds_a_freed_object(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return any(
            value is not None and _holds_a_freed_object(value) for value in node.values
        )
    if isinstance(node, ast.IfExp):
        return _holds_a_freed_object(node.body) or _holds_a_freed_object(node.orelse)
    if isinstance(node, ast.BoolOp):
        return any(_holds_a_freed_object(value) for value in node.values)
    if isinstance(node, ast.Lambda):
        # A default argument lives as long as the lambda does, so the lambda
        # counts as holding it wherever it is kept. One passed straight to a
        # call that keeps nothing of it, as shadow_gate keeps only the
        # verdict, is freed when the call returns.
        return any(
            default is not None and _holds_a_freed_object(default)
            for default in (*node.args.defaults, *node.args.kw_defaults)
        )
    if isinstance(node, ast.Call):
        name = _called_name(node)
        if name in _CONTAINER_CALLS:
            return any(_holds_a_freed_object(argument) for argument in _arguments(node))
        if name == "replace":
            # dataclasses.replace keeps every field it is not given, and
            # stores every field it is given.
            return any(_holds_a_freed_object(argument) for argument in _arguments(node))
        if name == "getattr" and len(node.args) >= 2:
            attribute = node.args[1]
            return (
                isinstance(attribute, ast.Constant)
                and attribute.value in _HOLDING_ATTRIBUTES
                and _holds_a_freed_object(node.args[0])
            )
    return False


def _alias_sites(function: ast.FunctionDef) -> list[tuple[int, str]]:
    """Where ``function`` makes a checked object reachable another way."""
    sites = []
    for node in ast.walk(function):
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
            and node.value is not None
            and _holds_a_freed_object(node.value)
        ):
            sites.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, (ast.For, ast.comprehension)) and _holds_a_freed_object(
            node.iter
        ):
            sites.append((node.iter.lineno, ast.unparse(node.iter)))
        elif isinstance(node, ast.withitem) and _holds_a_freed_object(
            node.context_expr
        ):
            sites.append((node.context_expr.lineno, ast.unparse(node.context_expr)))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _STORING_METHODS
            and any(_holds_a_freed_object(argument) for argument in _arguments(node))
        ):
            sites.append((node.lineno, ast.unparse(node)))
        elif (
            isinstance(node, ast.Call)
            and _called_name(node) == "setattr"
            and len(node.args) >= 3
            and _holds_a_freed_object(node.args[2])
        ):
            sites.append((node.lineno, ast.unparse(node)))
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node is not function
            and any(
                default is not None and _holds_a_freed_object(default)
                for default in (*node.args.defaults, *node.args.kw_defaults)
            )
        ):
            # A nested function is bound to a local, which keeps its default
            # arguments as long as the function object lives.
            sites.append((node.lineno, ast.unparse(node.args)))
    return sorted(sites)


def _main_function(builder) -> ast.FunctionDef:
    _, tree = _function_source(builder, "_main")
    (function,) = tree.body
    return function


def _name_lines(function: ast.AST, name: str, context: type) -> list[int]:
    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, context)
    )


def _deletes(function: ast.FunctionDef, name: str) -> list[ast.Delete]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]


def _single_top_level_delete(function: ast.FunctionDef, name: str) -> int:
    """The index in ``_main``'s body of its one ``del <name>`` statement."""
    (deleted,) = _deletes(function, name)
    # A top-level statement runs on every path that reaches it, not only
    # inside one branch.
    return function.body.index(deleted)


def _single_top_level_release(function: ast.FunctionDef, name: str) -> int:
    """The index in ``_main``'s body of its one ``<name> = None`` statement.

    ``_main`` frees an object that a shadow-gate lambda reads by rebinding its
    name to ``None``, because ruff reports a lambda's read of a name deleted
    later in the function as a read of an undefined name.
    """
    assert _deletes(function, name) == []
    (released,) = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and ast.unparse(node) == f"{name} = None"
    ]
    # A top-level statement runs on every path that reaches it, not only
    # inside one branch.
    return function.body.index(released)


#: The release tool's module-level functions, classes and variables, by name,
#: each with every statement that defines or rebinds it, in source order.
_ToolDefinitions = dict[str, list[ast.stmt]]


def _tool_definitions(builder) -> _ToolDefinitions:
    """The release tool's module-level definitions, with file line numbers."""
    tool = ast.parse(Path(builder.__file__).read_text())
    definitions: _ToolDefinitions = {}
    for node in tool.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # A variable such as a table of functions brings in what it names.
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions.setdefault(target.id, []).append(node)
    return definitions


def _tool_main(definitions: _ToolDefinitions) -> ast.FunctionDef:
    """``_main`` as the file defines it, with file line numbers."""
    (main,) = [
        node for node in definitions["_main"] if isinstance(node, ast.FunctionDef)
    ]
    return main


def _tool_definitions_reached(
    definitions: _ToolDefinitions, names: set[str]
) -> set[str]:
    """``names`` and every tool definition they name, transitively.

    A definition counts whether it is called or used as a value, such as a
    function put in a mapping of consumers and called through it later. A
    class brings in all its methods, and a variable what its value names.
    A name used only in a type annotation is not followed: an annotation
    runs nothing.
    """
    reached: set[str] = set()
    frontier = list(names)
    while frontier:
        name = frontier.pop()
        if name in reached:
            continue
        reached.add(name)
        for statement in definitions[name]:
            frontier.extend(_tool_definitions_named(statement, definitions))
    return reached


def _reached_after(
    definitions: _ToolDefinitions, later_statements: list[ast.stmt]
) -> set[str]:
    """The tool definitions that code after a point in ``_main`` can run.

    These are the definitions ``later_statements`` name, plus what code
    before the point can leave for later:
    - every tool class reachable from ``_main``, since an object built
      earlier, such as the build's progress reporter, can run its methods
      after the point;
    - every tool function ``_main`` uses as a value rather than calling it,
      which a local can hold until after the point;
    - what a call before the point names when ``_main`` binds its result to a
      local that it reads after the point, since that result (a closure, a
      function or an object) can run then.
    Each is followed through the definitions it names, and counts whole: a
    matrix read that such a function makes before it returns, while the
    matrix is still there, fails the scan too, and belongs in a function of
    its own.
    """
    main = _tool_main(definitions)
    named_later = set().union(
        *(
            _tool_definitions_named(statement, definitions)
            for statement in later_statements
        )
    )
    reachable_from_main = _tool_definitions_reached(
        definitions, _tool_definitions_named(main, definitions)
    )
    classes = {
        name
        for name in reachable_from_main
        if any(isinstance(statement, ast.ClassDef) for statement in definitions[name])
    }
    called = {
        id(node.func)
        for node in _walk_without_annotations(main)
        if isinstance(node, ast.Call)
    }
    functions_as_values = {
        node.id
        for node in _walk_without_annotations(main)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in called
        and any(
            isinstance(statement, ast.FunctionDef)
            for statement in definitions.get(node.id, ())
        )
    }
    later_nodes = {
        id(node) for statement in later_statements for node in ast.walk(statement)
    }
    read_later = {
        node.id
        for statement in later_statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    kept_for_later: set[str] = set()
    for node in ast.walk(main):
        if (
            id(node) in later_nodes
            or not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
            or not isinstance(node.value, ast.Call)
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        bound = {
            name.id
            for target in targets
            for name in ast.walk(target)
            if isinstance(name, ast.Name)
        }
        if bound & read_later:
            kept_for_later |= _tool_definitions_named(node.value, definitions)
    return _tool_definitions_reached(
        definitions,
        named_later | classes | functions_as_values | kept_for_later,
    )


def _walk_without_annotations(tree: ast.AST):
    """``ast.walk``, skipping argument, return and variable annotations."""
    pending = [tree]
    while pending:
        node = pending.pop()
        yield node
        for field, value in ast.iter_fields(node):
            if field in {"annotation", "returns"}:
                continue
            children = value if isinstance(value, list) else [value]
            pending.extend(child for child in children if isinstance(child, ast.AST))


def _tool_definitions_named(tree: ast.AST, definitions: _ToolDefinitions) -> set[str]:
    return {
        node.id
        for node in _walk_without_annotations(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in definitions
    }


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _execute(statements: list[ast.stmt], namespace: dict, label: str) -> None:
    module = ast.Module(body=statements, type_ignores=[])
    exec(compile(module, label, "exec"), namespace)


def test_main_binds_no_other_name_to_a_freed_object(builder) -> None:
    """A second name would keep an object alive after its ``del`` or drop."""
    main = _main_function(builder)
    sites = _alias_sites(main)
    assert tuple(statement for _, statement in sites) == _EXPECTED_ALIAS_BINDINGS


def test_the_alias_check_sees_each_way_of_keeping_an_object() -> None:
    """Each form below keeps ``base_frame`` or the matrix reachable."""
    keeping = (
        "kept = base_frame if flag else None",
        "kept = flag and base_frame",
        "kept = SimpleNamespace(frame=base_frame)",
        "kept = dataclasses.replace(ladder_outcome, support=support)",
        "kept = getattr(result, 'problem')",
        "kept = [result.problem.matrix]",
        "kept += [release_engine]",
        "frames.append(base_frame)",
        "for kept in (base_frame,):\n    pass",
        "with release_engine as kept:\n    pass",
        "kept = copy.copy(result)",
        "kept = functools.partial(write, result=result)",
        "kept = dataclasses.replace(outcome, result=result)",
        "setattr(holder, 'frame', base_frame)",
        "callbacks.append(lambda frame=base_frame: frame)",
        "kept = lambda frame=base_frame: frame",
        "def later(*, engine=release_engine):\n    pass",
        "kept = copy.deepcopy(result)",
        "kept = dict(frame=base_frame)",
        "kept = list((base_frame,))",
        "kept = set([release_engine])",
        "kept = tuple([base_frame])",
        "kept = frozenset({release_engine})",
        "kept: object = base_frame",
        "if (kept := base_frame) is not None:\n    pass",
        "kept = {'frame': base_frame}",
        "frames.add(base_frame)",
        "frames.extend([base_frame])",
        "frames.insert(0, base_frame)",
        "cache.setdefault('frame', base_frame)",
        "cache.update(frame=base_frame)",
        "kept = lambda *, frame=base_frame: frame",
        "kept = ladder_outcome.result.refit",
        "kept = result.selection",
    )
    harmless = (
        "total = base_frame.n('household')",
        "loss = result.final_loss",
        # shadow_gate keeps only the verdict, so its lambda is freed when it
        # returns.
        "shadow_gate('x', lambda frame=base_frame: gate(frame), context)",
        "kept = copy.copy(settings)",
        "setattr(holder, 'frame', other_frame)",
        "kept = dataclasses.replace(options, seed=seed)",
        "frames.append(base_frame.n('household'))",
    )
    for source, expected_sites in (
        *((source, 1) for source in keeping),
        *((source, 0) for source in harmless),
    ):
        function = ast.parse("def _main():\n" + textwrap.indent(source, "    "))
        assert len(_alias_sites(function.body[0])) == expected_sites, source


def test_main_keeps_only_the_receipt_of_the_acs_predictor_join(builder) -> None:
    """The join result is deleted before the next stage rebinds base_frame."""
    main = _main_function(builder)
    deleted_at = _single_top_level_delete(main, "acs_predictor_join_result")
    deleted_line = main.body[deleted_at].lineno
    reads = _name_lines(main, "acs_predictor_join_result", ast.Load)
    assert reads and max(reads) < deleted_line
    assert max(_name_lines(main, "acs_predictor_join_result", ast.Store)) < deleted_line
    # base_frame names the join's frame until the next stage rebinds it; the
    # result must be gone by then, or it keeps that frame alive.
    (frame_read,) = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and ast.unparse(node) == "base_frame = acs_predictor_join_result.frame"
    ]
    next_base_frame_binding = min(
        line for line in _name_lines(main, "base_frame", ast.Store) if line > frame_read
    )
    assert deleted_line < next_base_frame_binding
    # Only the receipt is read later.
    assert max(_name_lines(main, "acs_predictor_join_receipt", ast.Load)) > deleted_line


def test_main_frees_the_acs_join_frame_once_base_frame_is_rebound(builder) -> None:
    """Run ``_main``'s join statements: only base_frame holds the joined frame."""
    main = _main_function(builder)
    (join_at,) = [
        index
        for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "acs_predictor_join_result"
            for target in node.targets
        )
    ]
    deleted_at = _single_top_level_delete(main, "acs_predictor_join_result")
    # The fake hands its only reference to the joined frame to the result.
    unjoined = [_LargeObject()]
    joined_frame_reference = weakref.ref(unjoined[0])
    receipt = {"enabled": True, "join": {"matched_households": 3}}

    def join_acs_release_predictors(frame, **paths):
        return SimpleNamespace(frame=unjoined.pop(), receipt=receipt)

    namespace = {
        **vars(builder),
        "join_acs_release_predictors": join_acs_release_predictors,
        "base_frame": _LargeObject(),
        "args": SimpleNamespace(
            acs_person_zip=None,
            acs_person_sha256=None,
            acs_household_zip=None,
            acs_household_sha256=None,
        ),
        "telemetry": None,
    }
    _execute(
        main.body[join_at : deleted_at + 1], namespace, "<main ACS join statements>"
    )

    assert "acs_predictor_join_result" not in namespace
    assert namespace["acs_predictor_join_receipt"] == receipt
    assert namespace["base_frame"] is joined_frame_reference()
    # The next stage rebinds base_frame to its own output.
    namespace["base_frame"] = _LargeObject()
    gc.collect()
    assert joined_frame_reference() is None


def test_main_passes_each_donor_table_straight_into_its_stage(builder) -> None:
    """No local holds a donor table, so each is freed when its stage returns."""
    main = _main_function(builder)
    parents = {
        child: parent
        for parent in ast.walk(main)
        for child in ast.iter_child_nodes(parent)
    }
    loader_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and re.fullmatch(r"load_\w+_donor", node.func.id)
    ]
    assert sorted(call.func.id for call in loader_calls) == sorted(_DONOR_LOADERS)
    for call in loader_calls:
        keyword = parents[call]
        assert isinstance(keyword, ast.keyword), ast.unparse(call)
        stage = parents[keyword]
        assert isinstance(stage, ast.Call)
        assert isinstance(stage.func, ast.Name)
        assert stage.func.id.startswith("with_us_"), ast.unparse(stage.func)
    # A cached loader would keep its table for the process, whatever _main does.
    for loader in _DONOR_LOADERS:
        assert not hasattr(getattr(builder, loader), "cache_info"), loader


def test_main_frees_base_frame_after_the_export_input_mass_check(builder) -> None:
    """``base_frame = None`` follows the check that reads it last."""
    main = _main_function(builder)
    released_at = _single_top_level_release(main, "base_frame")
    released_line = main.body[released_at].lineno
    check = main.body[released_at - 1]
    assert isinstance(check, ast.Assign)
    assert _calls_named(check, "_record_export_input_mass_gate")
    # Every read, the shadow-gate lambdas' included, comes before the release.
    assert max(_name_lines(main, "base_frame", ast.Load)) < released_line
    assert max(_name_lines(main, "base_frame", ast.Store)) == released_line
    (write,) = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "write_dataset"
    ]
    assert released_line < write


def test_main_leaves_the_export_reference_frame_to_its_check(builder) -> None:
    """``_main`` binds no reference frame; the check loads it from the path."""
    main = _main_function(builder)
    assert _calls_named(main, "load_us_frame") == []
    assert "export_reference_frame" not in {
        node.id for node in ast.walk(main) if isinstance(node, ast.Name)
    }
    assert _calls_named(main, "_export_input_mass_gate") == []
    (call,) = _calls_named(main, "_record_export_input_mass_gate")
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords == {
        "reference_h5": "args.export_input_mass_reference_h5",
        "reference_name": "export_reference_name",
        "reference_record": "export_input_mass_reference",
        "relative_tolerance": "args.input_mass_relative_tolerance",
        "minimum_reference_total": "args.input_mass_minimum_reference_total",
        "terminal_gate_failures": "terminal_gate_failures",
    }


#: The export input-mass check's entry in the national gate battery.
_EXPORT_CHECK_ENTRY = "us_national_export_input_mass"
#: The result name ``_export_input_mass_gate`` gives, which the gate battery's
#: binding renames to the entry's declared gate.
_EXPORT_CHECK_RESULT_NAME = "input_mass_parity"
#: The options ``_main`` passes the export check, which passes them on to
#: ``_export_input_mass_gate``.
_CHECK_OPTIONS = {
    "relative_tolerance": 0.5,
    "minimum_reference_total": 1.0,
    "reference_name": "reference_populace_us_2024.h5",
}


class _ReferenceCheckStandIn:
    """Stands in for ``load_us_frame`` and ``_export_input_mass_gate``.

    Records each load, as its path and a weak reference to the frame it
    returns, and each call of the check, with ``reference`` saying which frame
    it got. It holds the export and base frames only through weak references,
    so a test can see whether the code under test keeps them. Setting
    ``failing`` to ``"load"`` or ``"check"`` makes that step raise.
    """

    def __init__(self, builder, monkeypatch, export_frame, base_frame) -> None:
        self.loaded: list[tuple[Path, weakref.ref]] = []
        self.gate_calls: list[dict[str, object]] = []
        self.failing: str | None = None
        self.verdict = builder.GateResult(
            name=_EXPORT_CHECK_RESULT_NAME, passed=True, details={"checked": True}
        )
        self._export_frame = weakref.ref(export_frame)
        self._base_frame = weakref.ref(base_frame)
        monkeypatch.setattr(builder, "load_us_frame", self.load_us_frame)
        monkeypatch.setattr(builder, "_export_input_mass_gate", self.gate)

    def load_us_frame(self, path):
        if self.failing == "load":
            raise OSError("the reference did not open")
        reference = _LargeObject()
        self.loaded.append((Path(path), weakref.ref(reference)))
        return reference

    def gate(self, candidate, base, *, reference_frame, **options):
        assert candidate is self._export_frame() and base is self._base_frame()
        if reference_frame is None:
            reference = None
        elif reference_frame is self.loaded[-1][1]():
            reference = "loaded"
        else:
            reference = "other"
        self.gate_calls.append({"reference": reference, **options})
        if self.failing == "check":
            raise RuntimeError("the gate failed")
        return self.verdict

    def frames_freed(self) -> bool:
        gc.collect()
        return self._export_frame() is None and self._base_frame() is None


@contextlib.contextmanager
def _armed_national_shadow(builder, monkeypatch, tmp_path):
    """Arm the national scope's gate battery shadow, as every real run does."""
    arm_us_tool_gate_shadows(monkeypatch)
    args = SimpleNamespace(
        out=tmp_path / "out",
        dry_run_gates_report=None,
        allow_input_mass_drift=False,
        allow_ecps_parity_gaps=False,
        allow_input_coverage_gaps=False,
        allow_esi_premium_gaps=False,
        allow_unpinned_feed=False,
        allow_qrf_tail_concentration=False,
        evidence_release=False,
        input_mass_relative_tolerance=_CHECK_OPTIONS["relative_tolerance"],
        input_mass_minimum_reference_total=_CHECK_OPTIONS["minimum_reference_total"],
    )
    builder._start_national_gate_shadow(args, release_id="microcosm-us-test")
    try:
        shadow = current_gate_shadow(_EXPORT_CHECK_ENTRY)
        assert shadow is not None
        yield shadow
    finally:
        builder._finish_national_gate_shadow(stopped_at="test teardown")


def _reference_h5(tmp_path) -> Path:
    path = tmp_path / "reference_populace_us_2024.h5"
    path.write_bytes(b"reference bytes")
    return path


def test_the_export_input_mass_check_frees_its_reference_when_it_returns(
    builder, monkeypatch, tmp_path
) -> None:
    """The check loads the reference itself and holds it only while it runs,
    whether it returns, raises, or records its error under earlier failures.
    It keeps neither the export frame nor base_frame."""
    frames = {"export": _LargeObject(), "base": _LargeObject()}
    stand_in = _ReferenceCheckStandIn(
        builder, monkeypatch, frames["export"], frames["base"]
    )
    reference_h5 = _reference_h5(tmp_path)
    gate_options = {
        **_CHECK_OPTIONS,
        "reviewed_exclusions": builder.US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
    }

    def check(reference_h5, *, failures):
        record: dict[str, object] = {"sha256": None}
        returned = builder._record_export_input_mass_gate(
            frames["export"],
            frames["base"],
            reference_h5=reference_h5,
            reference_record=record,
            terminal_gate_failures=failures,
            **_CHECK_OPTIONS,
        )
        return returned, record

    returned, record = check(reference_h5, failures=[])
    assert returned is stand_in.verdict
    assert record["sha256"] == builder._sha256(reference_h5)
    ((loaded_path, reference),) = stand_in.loaded
    assert loaded_path == reference_h5
    assert stand_in.gate_calls[0] == {"reference": "loaded", **gate_options}
    gc.collect()
    assert reference() is None

    # A check that raises on an otherwise-clean run frees its reference once
    # the error is handled.
    stand_in.failing = "check"
    with pytest.raises(RuntimeError, match="the gate failed") as raised:
        check(reference_h5, failures=[])
    # The traceback holds the check's frame, and so the reference, until the
    # error is released.
    del raised
    gc.collect()
    assert len(stand_in.loaded) == 2
    assert stand_in.gate_calls[1]["reference"] == "loaded"
    assert stand_in.loaded[1][1]() is None

    # Under earlier failures the error becomes one more failure line, and the
    # reference is freed when the check returns.
    failures = ["an earlier gate failed"]
    returned, _ = check(reference_h5, failures=failures)
    assert returned is None
    assert failures == [
        "an earlier gate failed",
        "Input mass parity failed: evaluation error under earlier gate failures: "
        "RuntimeError: the gate failed",
    ]
    gc.collect()
    assert len(stand_in.loaded) == 3 and stand_in.loaded[2][1]() is None

    # Without a reference H5 the check compares against base_frame and loads
    # nothing.
    stand_in.failing = None
    returned, record = check(None, failures=[])
    assert returned is stand_in.verdict and record["sha256"] is None
    assert len(stand_in.loaded) == 3
    assert stand_in.gate_calls[-1]["reference"] is None

    frames.clear()
    assert stand_in.frames_freed()


def test_the_shadow_sees_the_export_reference_and_does_not_keep_it(
    builder, monkeypatch, tmp_path
) -> None:
    """Every real run arms the gate battery's shadow, which evaluates the same
    check on an evidence context holding the loaded reference. The reference,
    the export frame and base_frame are still freed when the check returns."""
    frames = {"export": _LargeObject(), "base": _LargeObject()}
    stand_in = _ReferenceCheckStandIn(
        builder, monkeypatch, frames["export"], frames["base"]
    )
    with _armed_national_shadow(builder, monkeypatch, tmp_path) as shadow:
        returned = builder._record_export_input_mass_gate(
            frames["export"],
            frames["base"],
            reference_h5=_reference_h5(tmp_path),
            reference_record={"sha256": None},
            terminal_gate_failures=[],
            **_CHECK_OPTIONS,
        )
        outcome = shadow.outcome(_EXPORT_CHECK_ENTRY)
        differences = shadow.differences()

    assert returned is stand_in.verdict
    # Today's check and the shadow's evaluation each got the loaded frame.
    assert [call["reference"] for call in stand_in.gate_calls] == ["loaded", "loaded"]
    assert outcome is not None and outcome.status is GateStatus.PASSED
    assert differences == ()
    ((_, reference),) = stand_in.loaded
    gc.collect()
    assert reference() is None
    frames.clear()
    assert stand_in.frames_freed()


#: Each step the stand-in can make fail: the error it raises, and the line
#: the check appends for it under earlier failures.
_EXPORT_CHECK_ERRORS = {
    "load": (OSError, "OSError: the reference did not open"),
    "check": (RuntimeError, "RuntimeError: the gate failed"),
}


@pytest.mark.parametrize(
    "earlier_failures", [True, False], ids=["earlier-failures", "clean-run"]
)
@pytest.mark.parametrize("failing", ["load", "check"])
def test_the_shadow_records_an_export_check_error(
    builder, monkeypatch, tmp_path, failing, earlier_failures
) -> None:
    """A reference that does not load, or a check that raises, comes after
    the reference's sha256 is recorded. Under earlier failures it becomes one
    failure line; on an otherwise-clean run it propagates. The shadow records
    a check that raised once, as the same verdict as today's, and a reference
    that did not load as a check that did not run only when the build goes
    on: a build that stops there leaves the entry unreached."""
    error_type, error = _EXPORT_CHECK_ERRORS[failing]
    frames = {"export": _LargeObject(), "base": _LargeObject()}
    stand_in = _ReferenceCheckStandIn(
        builder, monkeypatch, frames["export"], frames["base"]
    )
    stand_in.failing = failing
    reference_h5 = _reference_h5(tmp_path)
    record: dict[str, object] = {"sha256": None}
    failures = ["an earlier gate failed"] if earlier_failures else []

    def check():
        return builder._record_export_input_mass_gate(
            frames["export"],
            frames["base"],
            reference_h5=reference_h5,
            reference_record=record,
            terminal_gate_failures=failures,
            **_CHECK_OPTIONS,
        )

    with _armed_national_shadow(builder, monkeypatch, tmp_path) as shadow:
        if earlier_failures:
            assert check() is None
        else:
            with pytest.raises(error_type) as raised:
                check()
            # The traceback holds the check's frame until it is released.
            del raised
        outcome = shadow.outcome(_EXPORT_CHECK_ENTRY)
        differences = shadow.differences()

    assert record["sha256"] == builder._sha256(reference_h5)
    assert failures == (
        [
            "an earlier gate failed",
            "Input mass parity failed: evaluation error under earlier gate "
            f"failures: {error}",
        ]
        if earlier_failures
        else []
    )
    if failing == "check":
        expected_status = GateStatus.FAILED
    else:
        expected_status = GateStatus.EVIDENCE_ABSENT if earlier_failures else None
    assert (outcome.status if outcome is not None else None) is expected_status
    assert differences == ()
    # Today's check and the shadow's evaluation each ran when the load did.
    assert len(stand_in.gate_calls) == (2 if failing == "check" else 0)
    gc.collect()
    assert all(reference() is None for _, reference in stand_in.loaded)
    frames.clear()
    assert stand_in.frames_freed()


def test_main_frees_the_release_engine_after_writing_the_dataset(builder) -> None:
    """The engine that writes the H5 is released once the written file's
    stored-input check has recorded its result, before the export audit and
    the calibration NPZ that end the build.

    The release does not sit between the write and that check: the stored-input
    test (test_us_stored_input_register) holds the check right after the
    write. The check reads only the written file's column names and the
    module-level PolicyEngine-US system, so the engine adds nothing to the
    memory the check uses."""
    main = _main_function(builder)
    released_at = _single_top_level_release(main, "release_engine")
    released_line = main.body[released_at].lineno
    # Every read, the shadow-gate lambdas' included, comes before the release.
    reads = _name_lines(main, "release_engine", ast.Load)
    assert reads and max(reads) < released_line
    (write,) = [
        node
        for node in main.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", None) == "write_dataset"
    ]
    assert ast.unparse(write.value.func.value) == "release_engine"
    written_file_check = main.body.index(write) + 1
    check_result_recorded = written_file_check + 1
    assert "_written_stored_input_verdict_mismatch" in ast.unparse(
        main.body[written_file_check]
    )
    assert isinstance(main.body[check_result_recorded], ast.If)
    assert ast.unparse(main.body[check_result_recorded].test) == (
        "stored_input_premise_failure is not None"
    )
    assert released_at == check_result_recorded + 1


def test_main_drops_the_constraint_matrix_before_rebinding_the_ladder_outcome(
    builder,
) -> None:
    """The matrix goes with the calibrated frames, before ladder_outcome's
    result is replaced, so neither name keeps it."""
    main = _main_function(builder)
    statements = [ast.unparse(node) for node in main.body]
    frames_dropped = next(
        index
        for index, text in enumerate(statements)
        if text.startswith("result = _without_calibrated_frames(")
    )
    matrix_dropped = statements.index("result = without_constraint_matrix(result)")
    ladder_rebound = next(
        index
        for index, node in enumerate(main.body)
        if index > frames_dropped
        and isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "isinstance(ladder_outcome, ExactKLadderCalibration)"
    )
    assert frames_dropped < matrix_dropped < ladder_rebound
    assert statements[matrix_dropped + 1 : ladder_rebound] == []
    # The solves, which need the matrix, run before the drop.
    calibration_calls = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None)
        in {"calibrate", "calibrate_l0_refit", "calibrate_exact_k_ladder"}
    ]
    assert calibration_calls
    assert max(calibration_calls) < main.body[matrix_dropped].lineno
    # After the drop, no tool code that can reach the calibration result reads
    # a matrix's entries: neither _main's later statements nor any tool
    # function or class they name, followed through the definitions those
    # name, reads ``.matrix`` or calls ``.estimates``. The calibrate package's
    # readers run on a dropped matrix in this module's output tests and in the
    # calibrate package's tests.
    definitions = _tool_definitions(builder)
    # The same statements as main's, with file line numbers.
    assert (
        ast.unparse(_tool_main(definitions).body[matrix_dropped])
        == (statements[matrix_dropped])
    )
    later_statements = _tool_main(definitions).body[matrix_dropped + 1 :]
    receivers = {
        call.func.id
        for statement in later_statements
        for call in ast.walk(statement)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in definitions
        # Passed the result itself or any part of it, such as result.problem.
        and any(
            isinstance(name, ast.Name) and name.id in {"result", "ladder_outcome"}
            for argument in _arguments(call)
            for name in ast.walk(argument)
        )
    }
    assert {
        "_release_gate_failures",
        "_write_npz",
        "_write_release_calibration_diagnostics",
        "_us_build_record",
        "_assert_export_matches_calibration",
    } <= receivers
    reached = _reached_after(definitions, later_statements)
    assert receivers <= reached
    trees: dict[str, list[ast.AST]] = {"_main": later_statements}
    for name in reached - {"_main"}:
        trees[name] = list(definitions[name])
    later_nodes = [
        (name, node)
        for name, roots in trees.items()
        for root in roots
        for node in ast.walk(root)
    ]
    # The placeholder keeps the matrix's shape and nonzero count.
    size_reads = {
        id(node.value)
        for _, node in later_nodes
        if isinstance(node, ast.Attribute) and node.attr in {"shape", "nnz"}
    }
    assert (
        sorted(
            f"{name}, line {node.lineno}: {ast.unparse(node)}"
            for name, node in later_nodes
            if (
                isinstance(node, ast.Attribute)
                and (
                    node.attr == "estimates"
                    or (node.attr == "matrix" and id(node) not in size_reads)
                )
            )
            or (
                isinstance(node, ast.Call)
                and _called_name(node) == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in {"matrix", "estimates"}
            )
        )
        == []
    )


def test_main_clears_the_coverage_cache_after_the_start_up_check(builder) -> None:
    """The tax-benefit system the start-up check caches is not kept for the run."""
    main = _main_function(builder)
    statements = [ast.unparse(node) for node in main.body]
    check = statements.index("assert_release_input_coverage_manifest_current()")
    assert statements[check + 1] == "clear_installed_list_parameter_baseline_cache()"
    assert statements.count("clear_installed_list_parameter_baseline_cache()") == 1
    # Nothing in the build fills the cache again: neither _main's later
    # statements nor any tool function or class they name, followed through
    # the definitions those name. The reform-coverage smoke, the cache's next
    # real reader, runs at certification, in its own process; the build
    # reaches it only through the scoring plan's request-recording run.
    fillers = {
        "assert_release_input_coverage_manifest_current",
        "resolve_probe_parameter_changes",
        "installed_list_parameter_baseline",
        "_installed_parameter_stages",
        # The package functions that reach installed_list_parameter_baseline.
        "us_reform_coverage_smoke_gate",
        "_build_reform",
    }
    definitions = _tool_definitions(builder)
    # The same statements as main's, with file line numbers.
    main_in_file = _tool_main(definitions)
    assert ast.unparse(main_in_file.body[check + 1]) == statements[check + 1]
    later_statements = main_in_file.body[check + 2 :]
    reached = _reached_after(definitions, later_statements)
    trees: dict[str, list[ast.AST]] = {"_main": later_statements}
    for name in reached - {"_main"}:
        trees[name] = list(definitions[name])

    def named_by(target: str) -> set[str]:
        """The reached code that names ``target``."""
        return {
            name
            for name, roots in trees.items()
            if any(
                target in _tool_definitions_named(root, definitions) for root in roots
            )
        }

    # Named by itself or through a module attribute, called or passed on (as
    # to functools.partial or a shadow-gate lambda).
    assert sorted(
        (name, node.id if isinstance(node, ast.Name) else node.attr)
        for name, roots in trees.items()
        for root in roots
        for node in ast.walk(root)
        if (isinstance(node, ast.Name) and node.id in fillers)
        or (isinstance(node, ast.Attribute) and node.attr in fillers)
    ) == [("_reform_coverage_smoke_consumer", "us_reform_coverage_smoke_gate")]
    # That consumer is reached only as one of the scoring plan's consumers,
    # which the plan runs before export recording requests only, building no
    # reform (test_us_post_export_scoring.py::
    # test_recording_the_smoke_plan_builds_no_reform_and_needs_no_engine).
    # The consumers run for real only at certification. A new build function
    # that names _post_export_consumers fails here until it is read: if it
    # only lists the consumers (their names, say) and calls none, add it to
    # the set below.
    assert named_by("_reform_coverage_smoke_consumer") == {"_post_export_consumers"}
    assert named_by("_post_export_consumers") == {"_record_post_export_scoring_plan"}
    real_scorers = {
        "_score_post_export_consumer",
        "_open_post_export_scorer",
        "_HouseholdBatchedPostExportScorer",
        "_BatchedPostExportReform",
    }
    assert real_scorers <= definitions.keys()
    assert real_scorers & reached == set()


def test_main_export_statements_leave_only_the_matrix_shape_and_nonzero_count(
    builder,
) -> None:
    """After ``_main``'s drop statements, each constraint matrix is replaced by
    its shape and nonzero count, and the target surface is unchanged."""
    from microcosm.calibrate import calibrate_l0_refit

    for full_pool in (True, False):
        target_frame, targets = _small_calibration_problem()
        result = (
            calibrate(target_frame, targets, epochs=3, mass="conserve", seed=0)
            if full_pool
            else calibrate_l0_refit(
                target_frame,
                targets,
                epochs=5,
                refit_epochs=5,
                target_records=20,
                seed=0,
            )
        )
        expected_surface = builder.target_surface_payload(result)
        matrices = [result.problem.matrix]
        if not full_pool:
            matrices.append(result.selection.problem.matrix)
        matrix_references = [weakref.ref(matrix) for matrix in matrices]
        del matrices
        ladder_outcome = builder.ExactKLadderCalibration(
            result=result,
            support=np.arange(len(result.weights)),
            selected_inclusion_probabilities=np.ones(len(result.weights)),
            selection_receipt={},
            refit_baseline_diagnostics={},
        )
        main = _main_function(builder)
        statements = [ast.unparse(node) for node in main.body]
        start = statements.index("result = without_constraint_matrix(result)")
        end = next(
            index
            for index, node in enumerate(main.body)
            if index > start
            and isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "isinstance(ladder_outcome, ExactKLadderCalibration)"
        )
        namespace = {
            **vars(builder),
            "result": result,
            "ladder_outcome": ladder_outcome,
        }
        del result, ladder_outcome
        _execute(
            main.body[start : end + 1], namespace, "<main constraint-matrix statements>"
        )
        gc.collect()
        assert [reference() for reference in matrix_references] == [None] * len(
            matrix_references
        )
        cleaned = namespace["result"]
        assert isinstance(cleaned.problem.matrix, DroppedConstraintMatrix)
        if not full_pool:
            assert isinstance(cleaned.selection.problem.matrix, DroppedConstraintMatrix)
        assert namespace["ladder_outcome"].result is cleaned
        assert builder.target_surface_payload(cleaned) == expected_surface


@pytest.mark.parametrize("full_pool", [True, False])
def test_dropping_the_constraint_matrix_leaves_the_written_calibration_unchanged(
    builder, tmp_path, full_pool
) -> None:
    """Writing from a result without its matrices raises nothing and gives the
    same calibration NPZ arrays, diagnostics payload, target surface and
    in-sample fit. Of these, only the target surface reads the matrix (its
    shape and nonzero count); the rest show that no reader needs it."""
    from microcosm.calibrate import (
        calibrate_l0_refit,
        diagnostics_payload,
        without_constraint_matrix,
    )

    target_frame, targets = _small_calibration_problem()
    result = (
        calibrate(target_frame, targets, epochs=3, mass="conserve", seed=0)
        if full_pool
        else calibrate_l0_refit(
            target_frame, targets, epochs=5, refit_epochs=5, target_records=20, seed=0
        )
    )
    registry = TargetRegistry(
        [
            TargetSpec(
                name="x_total@2024",
                entity="household",
                measure="x",
                value=1.0,
                source="fixture",
                family="fixture",
            )
        ],
        country="us",
    )
    dropped = without_constraint_matrix(result)

    builder._write_npz(tmp_path / "original.npz", result=result, registry=registry)
    builder._write_npz(tmp_path / "dropped.npz", result=dropped, registry=registry)
    with (
        np.load(tmp_path / "original.npz", allow_pickle=True) as original,
        np.load(tmp_path / "dropped.npz", allow_pickle=True) as without_matrix,
    ):
        assert original.files == without_matrix.files
        for name in original.files:
            assert original[name].dtype == without_matrix[name].dtype, name
            np.testing.assert_array_equal(original[name], without_matrix[name])
    assert json.dumps(
        diagnostics_payload(dropped), sort_keys=True, allow_nan=False
    ) == json.dumps(diagnostics_payload(result), sort_keys=True, allow_nan=False)
    assert builder.target_surface_payload(dropped) == builder.target_surface_payload(
        result
    )
    assert builder._in_sample_estimates(dropped) == builder._in_sample_estimates(result)
    assert builder._in_sample_targets(dropped) == builder._in_sample_targets(result)


def test_dropping_the_constraint_matrix_leaves_the_release_diagnostics_unchanged(
    builder, small_frame, tmp_path
) -> None:
    """The release's calibration_diagnostics.json, written the way _main writes
    it, is byte-identical with and without the constraint matrix."""
    from microcosm.calibrate import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
        without_constraint_matrix,
    )

    registry = TargetRegistry(
        (
            TargetSpec(
                name="income",
                entity="person",
                measure="income",
                value=500_000.0,
                source="fixture",
                hierarchy=CalibrationHierarchy(
                    provider=HierarchyNode("irs_soi", "IRS Statistics of Income"),
                    category=HierarchyCategory("irs_soi.income", "Income", "irs_soi"),
                    geography=HierarchyGeography("0400000US06", "California", "state"),
                    dimensions=(),
                    target=HierarchyNode("income", "Income"),
                ),
            ),
        ),
        country="us",
    )
    result = calibrate(small_frame, registry.to_target_set(), epochs=3, seed=0)
    passing_gate = builder.GateResult(
        name="passing", passed=True, details={"checked": True}
    )
    written = {}
    for label, calibrated in (
        ("original", result),
        ("dropped", without_constraint_matrix(result)),
    ):
        release_dir = tmp_path / label
        release_dir.mkdir()
        outcome = builder._write_release_calibration_diagnostics(
            result=calibrated,
            target_surface=builder.target_surface_payload(calibrated),
            release_dir=release_dir,
            registry=registry,
            base_dataset_sha256="0" * 64,
            compilation={"dropped_target_names": []},
            target_profile_gate=passing_gate,
            health_input_gate=passing_gate,
            base_population_gate=passing_gate,
            support_value_repairs={},
            audit_export_targets=False,
            gate_failures=[],
            default_dataset={"method": "dense_no_l0", "final_loss": result.final_loss},
        )
        assert outcome.status == "available", outcome
        written[label] = (release_dir / "calibration_diagnostics.json").read_bytes()
    assert written["dropped"] == written["original"]
