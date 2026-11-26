"""Which columns written between the release tool's repeated passes reach them.

``tools/build_us_fiscal_refresh_release.py`` computes December
``uncapped_ssi`` and ``is_medicaid_eligible`` for every person on the
pre-calibration frame and again on the calibrated export frame. These tests
intersect each variable's static dependency graph in the installed engine
(``variable_dependency_closure``: every variable its formulas can read, on any
branch and in any period) with the columns the stages between the two passes
write.

On policyengine-us 2.2.1:

* No column written between the two SSI passes is a node of December
  ``uncapped_ssi``'s graph, and the graph reads no weight and no population
  aggregate.
* Between the two Medicaid passes, the SNAP take-up stage's
  ``takes_up_snap_if_eligible`` and three outputs of the other-health-insurance
  stage are inputs of ``is_medicaid_eligible``. SNAP take-up reaches it only
  through the Medicaid work-requirement formulas, which ``is_medicaid_eligible``
  reads only when a work requirement applies, and none applies in 2024. The
  other-health outputs also reach it through formulas that run in 2024.

``engine_workflow/us/test_us_post_calibration_take_up_inputs.py`` runs the
release tool's passes on a small frame to confirm both results.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
    VariableDependencyClosure,
)
from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
)
from test_support.microcosm_build.us_post_calibration_take_up_inputs import (
    MEDICAID_PASS_VARIABLE,
    SSI_PASS_VARIABLE,
    StageOutputs,
    main_frame_assignments,
    stages_between_medicaid_passes,
    stages_between_ssi_passes,
)

#: The variables ``is_medicaid_eligible`` reads only inside
#: ``if <work requirement>.applies:`` blocks.
_WORK_REQUIREMENT_READS = (
    "ar_medicaid_work_requirement_eligible",
    "is_medicaid_work_requirement_applicable_adult",
    "medicaid_work_requirement_eligible",
)
#: Formulas the population-aggregate scan in
#: ``test_us_batched_target_materialization.py`` marks and its review accepts
#: as joins within one household: each person's claiming tax unit is in the
#: person's own household. A selection of whole households keeps them.
_HOUSEHOLD_CLAIMING_JOINS = (
    "medicaid_has_known_claiming_tax_unit",
    "medicaid_household_income",
    "medicaid_household_size",
)


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


@pytest.fixture(scope="module")
def metadata_index() -> PolicyEngineUSVariableMetadataIndex:
    return PolicyEngineUSVariableMetadataIndex()


def _closure_identity(closure: VariableDependencyClosure) -> tuple:
    return (
        closure.engine_version,
        len(closure.input_leaves),
        len(closure.formula_nodes),
        len(closure.edges),
        closure.sha256,
    )


def _graph_nodes_written(
    closure: VariableDependencyClosure, stages: tuple[StageOutputs, ...]
) -> dict[str, tuple[str, ...]]:
    nodes = set(closure.input_leaves) | set(closure.formula_nodes)
    return {stage.name: tuple(sorted(set(stage.columns) & nodes)) for stage in stages}


def _reachable(
    closure: VariableDependencyClosure,
    *,
    without_edges: set[tuple[str, str]],
) -> set[str]:
    dependencies: dict[str, set[str]] = defaultdict(set)
    for consumer, dependency in closure.edges:
        if (consumer, dependency) not in without_edges:
            dependencies[consumer].add(dependency)
    reached: set[str] = set()
    pending = [closure.root]
    while pending:
        variable = pending.pop()
        if variable in reached:
            continue
        reached.add(variable)
        pending.extend(dependencies[variable] - reached)
    return reached


def _applies_block_lines(source_path: Path) -> list[range]:
    """Line ranges of ``if <parameter>.applies:`` bodies in one source file."""

    tree = ast.parse(source_path.read_text())
    ranges = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "applies"
        ):
            ranges.append(range(node.body[0].lineno, node.body[-1].end_lineno + 1))
    return ranges


def test_release_tool_runs_these_stages_between_the_ssi_passes(builder) -> None:
    """The stage list the other tests use is the one ``_main`` runs.

    Every ``base_frame = f(...)`` between the first December SSI pass and the
    second names its stage, in build order, and the export frame is the
    calibrated-weight copy of ``base_frame``. A new stage between the passes
    fails here until its columns are checked.
    """

    assignments = main_frame_assignments(builder)
    between = [
        called
        for line, called in assignments.base_frame_calls
        if assignments.first_ssi_pass_line < line < assignments.second_ssi_pass_line
    ]

    assert between == [
        stage.builder_call for stage in stages_between_ssi_passes(builder)
    ]
    assert assignments.ssi_pass_frames == ("base_frame", "export_frame")
    export_calls = [
        (line, called)
        for line, called in assignments.export_frame_calls
        if line < assignments.second_ssi_pass_line
    ]
    assert sorted(called for _, called in export_calls) == [
        "_with_calibrated_weights",
        "_with_l0_refit_weights",
    ]
    last_stage_line = max(
        line
        for line, _ in assignments.base_frame_calls
        if line < assignments.second_ssi_pass_line
    )
    assert all(line > last_stage_line for line, _ in export_calls)
    assert [stage.builder_call for stage in stages_between_medicaid_passes()] == (
        between[between.index("_with_medicaid_take_up_outputs") :]
    )


def test_no_column_written_between_the_ssi_passes_reaches_december_uncapped_ssi(
    builder, metadata_index
) -> None:
    closure = metadata_index.variable_dependency_closure(SSI_PASS_VARIABLE)
    # A new engine graph needs this review again.
    assert _closure_identity(closure) == (
        "2.2.1",
        53,
        61,
        183,
        "703cfdf988d2fbecf05ea020f94d879e90f7553d449e165542afa070a86b6043",
    )

    stages = stages_between_ssi_passes(builder)
    assert _graph_nodes_written(closure, stages) == {stage.name: () for stage in stages}


def test_snap_take_up_and_other_health_outputs_are_medicaid_eligibility_inputs(
    metadata_index,
) -> None:
    closure = metadata_index.variable_dependency_closure(MEDICAID_PASS_VARIABLE)
    assert _closure_identity(closure) == (
        "2.2.1",
        481,
        2179,
        7733,
        "f771a669de6b5e94310b44f249eb6ab09f93fe4e4bd854eb64257311af08f17a",
    )

    assert _graph_nodes_written(closure, stages_between_medicaid_passes()) == {
        "medicaid_take_up": (),
        "snap_state_take_up": ("takes_up_snap_if_eligible",),
        "other_health_insurance": (
            "health_insurance_premiums",
            "health_insurance_premiums_without_medicare_part_b",
            "is_self_employed",
        ),
    }


def test_snap_take_up_reaches_medicaid_eligibility_only_through_work_requirements(
    builder, metadata_index
) -> None:
    """In 2024, SNAP take-up cannot change Medicaid eligibility; the
    other-health outputs still can.

    ``is_medicaid_eligible`` reads the three work-requirement variables only
    inside ``if <work requirement>.applies:`` blocks, and both parameters are
    false in the release year. Without those three reads the graph no longer
    reaches ``takes_up_snap_if_eligible``, and it still reaches every
    other-health output that the full graph reaches.
    """

    import policyengine_us
    from policyengine_us import CountryTaxBenefitSystem

    # Receipt paths are relative to the installed package directory.
    package_root = Path(policyengine_us.__file__).parent
    closure = metadata_index.variable_dependency_closure(MEDICAID_PASS_VARIABLE)
    for variable in _WORK_REQUIREMENT_READS:
        receipts = [
            receipt
            for receipt in metadata_index.consumer_receipts(variable)
            if receipt.consumer == MEDICAID_PASS_VARIABLE
        ]
        assert receipts, variable
        for receipt in receipts:
            applies_blocks = _applies_block_lines(package_root / receipt.path)
            assert any(receipt.line in block for block in applies_blocks), receipt

    parameters = CountryTaxBenefitSystem().parameters(f"{builder.PERIOD}-01-01")
    assert parameters.gov.hhs.medicaid.eligibility.work_requirements.applies is False
    assert parameters.gov.states.ar.dhs.medicaid.work_requirements.applies is False

    reached = _reachable(
        closure,
        without_edges={
            (MEDICAID_PASS_VARIABLE, variable) for variable in _WORK_REQUIREMENT_READS
        },
    )
    assert "takes_up_snap_if_eligible" not in reached
    assert {
        "health_insurance_premiums",
        "health_insurance_premiums_without_medicare_part_b",
        "is_self_employed",
    } <= reached


def test_neither_pass_variable_reads_weights_or_population_aggregates(
    builder, metadata_index
) -> None:
    """New weights leave both values unchanged, and so does dropping other
    households: no formula in either graph reads a weight or aggregates over
    the simulation's population."""

    for variable, household_joins in (
        (SSI_PASS_VARIABLE, set()),
        (MEDICAID_PASS_VARIABLE, set(_HOUSEHOLD_CLAIMING_JOINS)),
    ):
        closure = metadata_index.variable_dependency_closure(variable)
        nodes = set(closure.input_leaves) | set(closure.formula_nodes)
        assert not nodes & set(builder.US_POPULATION_AGGREGATE_VARIABLES), variable
        assert not {node for node in nodes if node.endswith("_weight")}, variable
        assert nodes & set(_HOUSEHOLD_CLAIMING_JOINS) == household_joins, variable
