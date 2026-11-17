"""The US release tool's ``_main`` frees large objects after their last use.

``_main`` is one long function, so a local it never deletes stays alive until
the run ends. These tests read the function's syntax tree, and execute slices
of its statements, to check that each large object is deleted, or never bound
to a local, once nothing reads it.
"""

# ruff: noqa: F403, F405
import re
import weakref

from microcosm.calibrate import DroppedConstraintMatrix
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
#: The statements in ``_main`` that bind another name, or a container, to one
#: of those objects or a large part of one, in source order. Any other such
#: binding would keep the object alive after its ``del`` or drop.
_EXPECTED_ALIAS_BINDINGS = (
    # The pre-selection frame on the frozen-support path. It is another object
    # than the base_frame deleted after the export check, and outlives it.
    "selection_candidate_frame = base_frame",
    # The join's frame becomes the working frame; the result is deleted.
    "base_frame = acs_predictor_join_result.frame",
    # Rebound to the matrix-free result before the export.
    "result = ladder_outcome.result",
)


class _LargeObject:
    """A stand-in for a frame or table that a weak reference can follow."""


def _holds_a_freed_object(node: ast.AST) -> bool:
    """Whether an expression evaluates to, or contains, a checked object."""
    if isinstance(node, ast.Name):
        return node.id in _FREED_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _HOLDING_ATTRIBUTES and _holds_a_freed_object(node.value)
    if isinstance(node, ast.Starred):
        return _holds_a_freed_object(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_holds_a_freed_object(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return any(
            value is not None and _holds_a_freed_object(value) for value in node.values
        )
    return False


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
    bindings = sorted(
        (node.lineno, ast.unparse(node))
        for node in ast.walk(main)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        and node.value is not None
        and _holds_a_freed_object(node.value)
    )
    assert tuple(statement for _, statement in bindings) == _EXPECTED_ALIAS_BINDINGS


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
    donor_names = {
        node.id
        for node in ast.walk(main)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id.endswith("_donor")
    }
    assert donor_names == set()
    # A cached loader would keep its table for the process, whatever _main does.
    for loader in _DONOR_LOADERS:
        assert not hasattr(getattr(builder, loader), "cache_info"), loader


def test_main_frees_base_frame_after_the_export_input_mass_check(builder) -> None:
    """``del base_frame`` follows the check that reads it last."""
    main = _main_function(builder)
    deleted_at = _single_top_level_delete(main, "base_frame")
    deleted_line = main.body[deleted_at].lineno
    check = main.body[deleted_at - 1]
    assert isinstance(check, ast.Try)
    assert _calls_named(check, "_export_input_mass_gate_with_reference_h5")
    assert max(_name_lines(main, "base_frame", ast.Load)) < deleted_line
    assert max(_name_lines(main, "base_frame", ast.Store)) < deleted_line
    (write,) = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "write_dataset"
    ]
    assert deleted_line < write


def test_main_leaves_the_export_reference_frame_to_its_check(builder) -> None:
    """``_main`` binds no reference frame; the check loads it from the path."""
    main = _main_function(builder)
    assert _calls_named(main, "load_us_frame") == []
    assert "export_reference_frame" not in {
        node.id for node in ast.walk(main) if isinstance(node, ast.Name)
    }
    assert _calls_named(main, "_export_input_mass_gate") == []
    (call,) = _calls_named(main, "_export_input_mass_gate_with_reference_h5")
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords["reference_h5"] == "args.export_input_mass_reference_h5"
    assert keywords["reference_name"] == "export_reference_name"


def _check_options(builder) -> dict[str, object]:
    return {
        "relative_tolerance": 0.05,
        "minimum_reference_total": 1.0,
        "reference_name": "reference_populace_us_2024.h5",
        "reviewed_exclusions": builder.US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
    }


def test_the_export_input_mass_check_frees_its_reference_when_it_returns(
    builder, monkeypatch
) -> None:
    """The check loads the reference itself and holds it only while it runs."""
    loaded: list[tuple[Path, weakref.ref]] = []
    gate_calls: list[dict[str, object]] = []
    export_frame, base_frame = _LargeObject(), _LargeObject()
    verdict = builder.GateResult(
        name="export_input_mass_parity", passed=True, details={"checked": True}
    )

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
    reference_h5 = Path("reference_populace_us_2024.h5")

    returned = builder._export_input_mass_gate_with_reference_h5(
        export_frame, base_frame, reference_h5=reference_h5, **_check_options(builder)
    )
    assert returned is verdict
    ((loaded_path, reference),) = loaded
    assert loaded_path == reference_h5
    assert gate_calls[0] == {"reference": "loaded", **_check_options(builder)}
    gc.collect()
    assert reference() is None

    # A check that raises frees its reference once the error is handled.
    failing = {**_check_options(builder), "relative_tolerance": -1.0}
    with pytest.raises(RuntimeError, match="the gate failed") as raised:
        builder._export_input_mass_gate_with_reference_h5(
            export_frame, base_frame, reference_h5=reference_h5, **failing
        )
    # The traceback holds the check's frame, and so the reference, until the
    # error is released, as _main's except clause does at its end.
    del raised
    gc.collect()
    assert len(loaded) == 2 and gate_calls[1]["reference"] == "loaded"
    assert loaded[1][1]() is None

    # Without a reference H5 the check compares against base_frame and loads
    # nothing.
    assert (
        builder._export_input_mass_gate_with_reference_h5(
            export_frame, base_frame, reference_h5=None, **_check_options(builder)
        )
        is verdict
    )
    assert len(loaded) == 2
    assert gate_calls[-1]["reference"] is None


def test_main_frees_the_release_engine_after_writing_the_dataset(builder) -> None:
    """The engine that writes the H5 is deleted before post-export scoring."""
    main = _main_function(builder)
    deleted_at = _single_top_level_delete(main, "release_engine")
    deleted_line = main.body[deleted_at].lineno
    reads = _name_lines(main, "release_engine", ast.Load)
    assert reads and max(reads) < deleted_line
    (write,) = [
        node
        for node in main.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", None) == "write_dataset"
    ]
    assert ast.unparse(write.value.func.value) == "release_engine"
    assert write.lineno < deleted_line
    (scorer,) = _calls_named(main, "_open_post_export_scorer")
    assert deleted_line < scorer.lineno


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
    # After the drop, the release tool's own code never reads a matrix's
    # entries: nothing in it reads ``.matrix`` or calls ``.estimates``. The
    # calibrate package's readers are tested with a dropped matrix there.
    tool = ast.parse(Path(builder.__file__).read_text())
    assert [
        f"{node.lineno}: {ast.unparse(node)}"
        for node in ast.walk(tool)
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
    for filler in (
        "assert_release_input_coverage_manifest_current",
        "resolve_probe_parameter_changes",
        "installed_list_parameter_baseline",
        "_installed_parameter_stages",
    ):
        assert all(call.lineno < cleared_line for call in _calls_named(main, filler))


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
    """The calibration NPZ, the diagnostics payload and the target surface
    written from a result without its matrices equal those of the original."""
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
