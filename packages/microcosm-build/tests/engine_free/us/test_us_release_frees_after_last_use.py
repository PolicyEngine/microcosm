"""The US release tool's ``_main`` frees large objects after their last use.

``_main`` is one long function, so a local it never deletes stays alive until
the run ends. These tests read the function's syntax tree, and execute slices
of its statements, to check that each large object is deleted, rebound to
``None``, or never bound to a local, once nothing reads it.
"""

# ruff: noqa: F403, F405
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
#: Calls whose result holds their arguments.
_CONTAINER_CALLS = frozenset(
    {"SimpleNamespace", "dict", "frozenset", "list", "set", "tuple"}
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
    if isinstance(node, ast.Call):
        name = _called_name(node)
        if name in _CONTAINER_CALLS:
            return any(_holds_a_freed_object(argument) for argument in _arguments(node))
        if name == "replace":
            # dataclasses.replace keeps every field it is not given.
            return bool(node.args) and _holds_a_freed_object(node.args[0])
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
    )
    harmless = ("total = base_frame.n('household')", "loss = result.final_loss")
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


def _fake_reference_check(builder, monkeypatch, export_frame, base_frame, verdict):
    """Stand in ``load_us_frame`` and ``_export_input_mass_gate``.

    Returns the loads, as each path with a weak reference to its frame, and
    each call of the gate, with ``reference`` saying which frame it got.
    """
    loaded: list[tuple[Path, weakref.ref]] = []
    gate_calls: list[dict[str, object]] = []

    def load_us_frame(path):
        reference = _LargeObject()
        loaded.append((Path(path), weakref.ref(reference)))
        return reference

    def gate(candidate, base, *, reference_frame, **options):
        assert candidate is export_frame and base is base_frame
        if reference_frame is None:
            reference = None
        elif reference_frame is loaded[-1][1]():
            reference = "loaded"
        else:
            reference = "other"
        gate_calls.append({"reference": reference, **options})
        if options["relative_tolerance"] < 0:
            raise RuntimeError("the gate failed")
        return verdict

    monkeypatch.setattr(builder, "load_us_frame", load_us_frame)
    monkeypatch.setattr(builder, "_export_input_mass_gate", gate)
    return loaded, gate_calls


def test_the_export_input_mass_check_frees_its_reference_when_it_returns(
    builder, monkeypatch, tmp_path
) -> None:
    """The check loads the reference itself and holds it only while it runs,
    whether it returns, raises, or records its error under earlier failures."""
    export_frame, base_frame = _LargeObject(), _LargeObject()
    verdict = builder.GateResult(
        name=_EXPORT_CHECK_RESULT_NAME, passed=True, details={"checked": True}
    )
    loaded, gate_calls = _fake_reference_check(
        builder, monkeypatch, export_frame, base_frame, verdict
    )
    reference_h5 = tmp_path / "reference_populace_us_2024.h5"
    reference_h5.write_bytes(b"reference bytes")
    gate_options = {
        **_CHECK_OPTIONS,
        "reviewed_exclusions": builder.US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
    }

    def check(reference_h5, *, failures, **overrides):
        record: dict[str, object] = {"sha256": None}
        returned = builder._record_export_input_mass_gate(
            export_frame,
            base_frame,
            reference_h5=reference_h5,
            reference_record=record,
            terminal_gate_failures=failures,
            **{**_CHECK_OPTIONS, **overrides},
        )
        return returned, record

    returned, record = check(reference_h5, failures=[])
    assert returned is verdict
    assert record["sha256"] == builder._sha256(reference_h5)
    ((loaded_path, reference),) = loaded
    assert loaded_path == reference_h5
    assert gate_calls[0] == {"reference": "loaded", **gate_options}
    gc.collect()
    assert reference() is None

    # A check that raises on an otherwise-clean run frees its reference once
    # the error is handled.
    with pytest.raises(RuntimeError, match="the gate failed") as raised:
        check(reference_h5, failures=[], relative_tolerance=-1.0)
    # The traceback holds the check's frame, and so the reference, until the
    # error is released.
    del raised
    gc.collect()
    assert len(loaded) == 2 and gate_calls[1]["reference"] == "loaded"
    assert loaded[1][1]() is None

    # Under earlier failures the error becomes one more failure line, and the
    # reference is freed when the check returns.
    failures = ["an earlier gate failed"]
    returned, _ = check(reference_h5, failures=failures, relative_tolerance=-1.0)
    assert returned is None
    assert failures == [
        "an earlier gate failed",
        "Input mass parity failed: evaluation error under earlier gate failures: "
        "RuntimeError: the gate failed",
    ]
    gc.collect()
    assert len(loaded) == 3 and loaded[2][1]() is None

    # Without a reference H5 the check compares against base_frame and loads
    # nothing.
    returned, record = check(None, failures=[])
    assert returned is verdict and record["sha256"] is None
    assert len(loaded) == 3
    assert gate_calls[-1]["reference"] is None


def test_the_shadow_sees_the_export_reference_and_does_not_keep_it(
    builder, monkeypatch, tmp_path
) -> None:
    """Every real run arms the gate battery's shadow, which evaluates the same
    check on an evidence context holding the loaded reference. The reference
    is still freed when the check returns."""
    arm_us_tool_gate_shadows(monkeypatch)
    export_frame, base_frame = _LargeObject(), _LargeObject()
    verdict = builder.GateResult(
        name=_EXPORT_CHECK_RESULT_NAME, passed=True, details={"checked": True}
    )
    loaded, gate_calls = _fake_reference_check(
        builder, monkeypatch, export_frame, base_frame, verdict
    )
    reference_h5 = tmp_path / "reference_populace_us_2024.h5"
    reference_h5.write_bytes(b"reference bytes")
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
        shadow = current_gate_shadow("us_national_export_input_mass")
        assert shadow is not None
        returned = builder._record_export_input_mass_gate(
            export_frame,
            base_frame,
            reference_h5=reference_h5,
            reference_record={"sha256": None},
            terminal_gate_failures=[],
            **_CHECK_OPTIONS,
        )
        outcome = shadow.outcome("us_national_export_input_mass")
        differences = shadow.differences()
    finally:
        builder._finish_national_gate_shadow(stopped_at="test teardown")

    assert returned is verdict
    # Today's check and the shadow's evaluation each got the loaded frame.
    assert [call["reference"] for call in gate_calls] == ["loaded", "loaded"]
    assert outcome is not None and outcome.status is GateStatus.PASSED
    assert differences == ()
    ((_, reference),) = loaded
    gc.collect()
    assert reference() is None


def test_main_frees_the_release_engine_after_writing_the_dataset(builder) -> None:
    """The engine that writes the H5 is released right after the write, before
    the written-file check, the export audit and the calibration NPZ that end
    the build."""
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
    assert released_at == main.body.index(write) + 1


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
    # function they pass result or ladder_outcome to, followed through the
    # tool's own calls, reads ``.matrix`` or calls ``.estimates``. The
    # calibrate package's readers run on a dropped matrix in this module's
    # output tests and in the calibrate package's tests.
    dropped_line = main.body[matrix_dropped].lineno
    tool = ast.parse(Path(builder.__file__).read_text())
    functions = {
        node.name: node for node in tool.body if isinstance(node, ast.FunctionDef)
    }
    frontier = [
        call.func.id
        for call in ast.walk(main)
        if isinstance(call, ast.Call)
        and call.lineno > dropped_line
        and isinstance(call.func, ast.Name)
        and call.func.id in functions
        and any(
            isinstance(argument, ast.Name)
            and argument.id in {"result", "ladder_outcome"}
            for argument in _arguments(call)
        )
    ]
    assert {
        "_release_gate_failures",
        "_write_npz",
        "_write_release_calibration_diagnostics",
    } <= set(frontier)
    reached: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in reached:
            continue
        reached.add(name)
        frontier.extend(
            call.func.id
            for call in ast.walk(functions[name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in functions
        )
    later_nodes = [
        node for node in ast.walk(main) if getattr(node, "lineno", 0) > dropped_line
    ]
    for name in sorted(reached - {"_main"}):
        later_nodes.extend(ast.walk(functions[name]))
    assert [
        f"{node.lineno}: {ast.unparse(node)}"
        for node in later_nodes
        if isinstance(node, ast.Attribute) and node.attr in {"matrix", "estimates"}
    ] == []


def test_main_clears_the_coverage_cache_after_the_start_up_check(builder) -> None:
    """The tax-benefit system the start-up check caches is not kept for the run."""
    main = _main_function(builder)
    statements = [ast.unparse(node) for node in main.body]
    check = statements.index("assert_release_input_coverage_manifest_current()")
    assert statements[check + 1] == "clear_installed_list_parameter_baseline_cache()"
    assert statements.count("clear_installed_list_parameter_baseline_cache()") == 1
    # Nothing in _main fills the cache again; the reform-coverage smoke, after
    # the H5 write, is the next reader.
    cleared_line = main.body[check + 1].lineno
    fillers = {
        "assert_release_input_coverage_manifest_current",
        "resolve_probe_parameter_changes",
        "installed_list_parameter_baseline",
        "_installed_parameter_stages",
    }
    # Called by name or through a module attribute.
    assert [
        f"{call.lineno}: {ast.unparse(call)}"
        for call in ast.walk(main)
        if isinstance(call, ast.Call)
        and _called_name(call) in fillers
        and call.lineno > cleared_line
    ] == []


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
