"""Production propagation contract for neutral US immigration facts."""

from __future__ import annotations

import ast
import inspect

import microcosm.build.us_runtime as us_runtime
import microcosm.build.us_runtime.immigration as immigration_runtime
from microcosm.build.us_runtime.acs_transfer import (
    acs_transfer_execution_contract_identity,
)
from microcosm.build.us_runtime.immigration import (
    US_IMMIGRATION_ALL_OUTPUT_COLUMNS,
    US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS,
    us_immigration_stage_spec,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_OPERATOR_CONTRACTS,
    pool_input_surface,
    pool_transfer_target_families,
)
from microcosm.build.us_runtime.operator_boundary import (
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_SOURCE_OUTPUTS,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    US_LATE_SOURCE_INPUT_INVENTORIES,
)


def _transfer_targets() -> set[str]:
    return {
        target
        for entity_families in pool_transfer_target_families().values()
        for targets in entity_families.values()
        for target in targets
    }


def test_neutral_immigration_facts_cross_the_whole_pool_boundary() -> None:
    expected = frozenset(US_IMMIGRATION_ALL_OUTPUT_COLUMNS)

    assert PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["immigration"]["person"] == expected
    assert (
        POOL_OPERATOR_CONTRACTS["with_us_immigration_inputs"].execution_scope
        == "whole_pool"
    )
    producer_outputs = CANONICAL_US_LATE_SOURCE_OUTPUTS["with_us_immigration_inputs"]
    assert {(output.entity, output.column) for output in producer_outputs} == {
        ("person", column) for column in expected
    }
    assert {output.coverage_scope for output in producer_outputs} == {"whole_pool"}


def test_immigration_is_not_a_qrf_transfer_or_reconciliation_family() -> None:
    immigration_outputs = set(US_IMMIGRATION_ALL_OUTPUT_COLUMNS)

    assert not immigration_outputs.intersection(_transfer_targets())
    assert all(
        group.family != "source_operator_immigration"
        for group in CANONICAL_US_LATE_TRANSFER_GROUPS
    )
    assert len(CANONICAL_US_LATE_TRANSFER_GROUPS) == 18

    execution = acs_transfer_execution_contract_identity(targets=())
    assert not any("immigration" in key for key in execution)
    assert "humanitarian_immigration" not in execution["post_transfer_structure"]


def test_policyengine_bridge_columns_are_derived_inputs_only() -> None:
    surface = {entry.variable: entry for entry in pool_input_surface()}

    for column in US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS:
        assert surface[column].provenance == ("whole_pool_source_operator",)
    assert not (
        set(US_IMMIGRATION_ALL_OUTPUT_COLUMNS)
        - set(US_IMMIGRATION_LEGACY_OUTPUT_COLUMNS)
    ).intersection(surface)


def test_immigration_source_stage_is_target_free_census_evidence() -> None:
    stage = us_immigration_stage_spec()
    derive = next(
        operation
        for operation in stage.operations
        if operation.kind == "derive_immigration_status"
    )

    assert set(stage.outputs) == set(US_IMMIGRATION_ALL_OUTPUT_COLUMNS)
    assert derive.parameters == {
        "seed_from_build_config": True,
        "time_period_from_build_config": True,
    }
    assert {artifact["format"] for artifact in stage.artifacts} == {
        "census_asec",
        "census_acs_pums",
    }
    assert all("census.gov" in artifact["source"] for artifact in stage.artifacts)


def test_immigration_source_inventory_binds_every_literal_survey_read() -> None:
    source = ast.parse(inspect.getsource(immigration_runtime))
    source_read_helpers = {
        "_integer_column",
        "_float_column",
        "_numeric_evidence_column",
        "_source_code_is",
        "_positive_source_value",
    }
    reads = {
        node.args[1].value
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in source_read_helpers
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }
    inventory = US_LATE_SOURCE_INPUT_INVENTORIES["with_us_immigration_inputs"]
    declared = {
        column.column
        for requirement in inventory.requirements
        for alternative in requirement.alternatives
        for column in alternative
        if column.entity == "person"
    }

    assert {"SSP", "SSIP", "A_AGE", "age"} <= reads
    assert reads <= declared


def test_target_anchored_assignment_api_is_retired() -> None:
    for module in (us_runtime, immigration_runtime):
        for name in (
            "ImmigrationControls",
            "UndocumentedControls",
            "HumanitarianDraw",
            "us_immigration_controls",
            "reconcile_us_immigration_humanitarian_transfer",
            "us_immigration_humanitarian_draw_mask",
        ):
            assert not hasattr(module, name), (module.__name__, name)
