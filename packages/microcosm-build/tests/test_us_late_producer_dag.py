"""Late-stage producer DAG doctrine regressions (microcosm#653)."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from microcosm.build.us_runtime.late_producer_dag import (
    ProducerContract,
    ProducerInput,
    ProducerOutput,
    derive_producer_schedule,
    run_producer_when_ready,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_PRODUCER_REGISTRY,
    CANONICAL_US_LATE_PRODUCER_SCHEDULE,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    US_LATE_EXTERNAL_STAGES,
    US_LATE_PRIMARY_PUF_STAGE,
    US_LATE_SOURCE_FINALIZER_STAGE,
    US_LATE_SOURCE_INPUT_INVENTORIES,
    source_producer_name,
    transfer_producer_name,
    us_late_producer_schedule_receipt,
)


def _contract(name: str, *dependencies: str) -> ProducerContract:
    return ProducerContract(
        name=name,
        kind="fixture",
        inputs=tuple(
            ProducerInput(
                entity="person",
                column=f"{dependency}_output",
                required_scope="whole_pool",
                producing_stage=dependency,
            )
            for dependency in dependencies
        ),
        outputs=(
            ProducerOutput(
                entity="person",
                column=f"{name}_output",
                coverage_scope="whole_pool",
            ),
        ),
    )


def test_unfilled_late_input_refuses_before_producer_runs() -> None:
    requirement = ProducerInput(
        entity="person",
        column="late_input",
        required_scope="cps_projection",
        producing_stage="transfer:person/puf_tax_itemization__batch_5",
    )
    consumer = ProducerContract(
        name="with_fixture_consumer",
        kind="source_operator",
        inputs=(requirement,),
        outputs=(),
    )
    invoked = False

    def callback() -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)with_fixture_consumer.*person\.late_input.*1 unfilled.*"
            r"cps_projection.*transfer:person/puf_tax_itemization__batch_5"
        ),
    ):
        run_producer_when_ready(
            consumer,
            callback,
            unfilled_rows={requirement: 1},
            invalid_rows={requirement: 0},
            absence_receipts={},
        )

    assert invoked is False


def test_declared_absence_never_tolerates_invalid_input() -> None:
    receipt_id = "optional_input:consumer:predictor"
    requirement = ProducerInput(
        entity="person",
        column="@effective:predictor",
        required_scope="whole_pool",
        producing_stage="post_clone_input_surface",
        tolerated_absence_receipts=(receipt_id,),
    )
    consumer = ProducerContract(
        name="consumer",
        kind="fixture",
        inputs=(requirement,),
        outputs=(),
    )
    forged_absence = {
        receipt_id: {
            "receipt_id": receipt_id,
            "status": "declared_absence",
            "entity": "person",
            "column": "@effective:predictor",
            "required_scope": "whole_pool",
            "rows": 1,
        }
    }

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)consumer.*person\.@effective:predictor.*1 invalid.*"
            r"post_clone_input_surface"
        ),
    ):
        run_producer_when_ready(
            consumer,
            lambda: pytest.fail("invalid input reached callback"),
            unfilled_rows={requirement: 0},
            invalid_rows={requirement: 1},
            absence_receipts=forged_absence,
        )


def test_readiness_requires_exact_declared_count_surfaces() -> None:
    requirement = ProducerInput(
        entity="person",
        column="required_input",
        required_scope="whole_pool",
        producing_stage="producer",
    )
    consumer = ProducerContract("consumer", "fixture", (requirement,), ())

    with pytest.raises(
        ValueError,
        match=r"consumer.*unfilled-row readiness.*missing=.*required_input",
    ):
        run_producer_when_ready(
            consumer,
            lambda: pytest.fail("omitted input reached callback"),
            unfilled_rows={},
            invalid_rows={requirement: 0},
            absence_receipts={},
        )

    with pytest.raises(
        ValueError,
        match=r"consumer.*invalid-value readiness.*missing=.*required_input",
    ):
        run_producer_when_ready(
            consumer,
            lambda: pytest.fail("omitted input reached callback"),
            unfilled_rows={requirement: 0},
            invalid_rows={},
            absence_receipts={},
        )


def test_synthetic_producer_cycle_is_rejected_with_named_cycle() -> None:
    registry = {
        "alpha": _contract("alpha", "charlie"),
        "bravo": _contract("bravo", "alpha"),
        "charlie": _contract("charlie", "bravo"),
    }

    with pytest.raises(
        RuntimeError,
        match=r"alpha -> bravo -> charlie -> alpha",
    ):
        derive_producer_schedule(registry)


def test_derived_schedule_is_byte_stable_under_registry_iteration_order() -> None:
    contracts = (
        _contract("alpha"),
        _contract("bravo"),
        _contract("charlie", "alpha", "bravo"),
        _contract("delta", "charlie"),
    )
    forward = OrderedDict((contract.name, contract) for contract in contracts)
    reverse = OrderedDict((contract.name, contract) for contract in reversed(contracts))

    forward_schedule = derive_producer_schedule(forward)
    reverse_schedule = derive_producer_schedule(reverse)

    assert forward_schedule.order == reverse_schedule.order
    assert forward_schedule.waves == reverse_schedule.waves
    assert forward_schedule.edges == reverse_schedule.edges
    assert forward_schedule.canonical_json == reverse_schedule.canonical_json
    assert forward_schedule.sha256 == reverse_schedule.sha256


def test_canonical_us_late_registry_has_exact_producer_surface() -> None:
    registry = CANONICAL_US_LATE_PRODUCER_REGISTRY
    groups = CANONICAL_US_LATE_TRANSFER_GROUPS

    assert len(registry) == 37
    assert len(groups) == 19
    assert sum(len(group.targets) for group in groups) == 70
    assert {contract.kind for contract in registry.values()} == {
        "primary_puf",
        "post_clone_source",
        "late_transfer",
        "source_finalizer",
    }
    assert len(registry[US_LATE_PRIMARY_PUF_STAGE].inputs) == 15
    primary_outputs = registry[US_LATE_PRIMARY_PUF_STAGE].outputs
    assert len(primary_outputs) == 66
    assert sum(output.coverage_scope == "puf_clone" for output in primary_outputs) == 65
    assert {
        (output.entity, output.column, output.coverage_scope)
        for output in primary_outputs
        if output.coverage_scope == "whole_pool"
    } == {("person", "person_support_clone_index", "whole_pool")}
    assert all(contract.inputs for contract in registry.values())
    assert {
        name
        for name, contract in registry.items()
        if contract.kind == "post_clone_source"
    } == {
        source_producer_name(operator)
        for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    }
    assert {
        name for name, contract in registry.items() if contract.kind == "late_transfer"
    } == {group.name for group in groups}
    for group in groups:
        assert {
            (output.entity, output.column)
            for output in CANONICAL_US_LATE_PRODUCER_REGISTRY[group.name].outputs
        } == {(group.entity, target) for target in group.targets}


def test_canonical_us_late_registry_declares_required_cross_producer_edges() -> None:
    edges = set(CANONICAL_US_LATE_PRODUCER_SCHEDULE.edges)

    assert len(edges) == 70
    assert (
        source_producer_name("with_us_pregnancy_inputs"),
        source_producer_name("with_us_wic_claim_input"),
    ) in edges
    assert (
        source_producer_name("with_us_childcare_inputs"),
        source_producer_name("with_us_adult_care_inputs"),
    ) in edges
    assert (
        transfer_producer_name("person", "puf_tax_itemization__batch_5"),
        source_producer_name("with_us_adult_care_inputs"),
    ) in edges
    assert (
        transfer_producer_name("person", "puf_tax_itemization__batch_2"),
        source_producer_name("with_us_education_inputs"),
    ) in edges
    assert {
        consumer
        for producer, consumer in edges
        if producer == US_LATE_PRIMARY_PUF_STAGE and consumer.startswith("transfer:")
    } == {group.name for group in CANONICAL_US_LATE_TRANSFER_GROUPS}
    assert {
        consumer
        for producer, consumer in edges
        if producer == US_LATE_PRIMARY_PUF_STAGE and consumer.startswith("source:")
    } == {
        source_producer_name(operator)
        for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    }
    assert CANONICAL_US_LATE_PRODUCER_SCHEDULE.waves[0] == (US_LATE_PRIMARY_PUF_STAGE,)
    assert (
        US_LATE_SOURCE_FINALIZER_STAGE
        in (CANONICAL_US_LATE_PRODUCER_SCHEDULE.waves[-1])
    )
    assert {
        producer
        for producer, consumer in edges
        if consumer == US_LATE_SOURCE_FINALIZER_STAGE
    } == {
        source_producer_name(operator)
        for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    }
    assert {
        (output.entity, output.column, output.coverage_scope)
        for output in CANONICAL_US_LATE_PRODUCER_REGISTRY[
            US_LATE_SOURCE_FINALIZER_STAGE
        ].outputs
    } == {
        ("person", column, "whole_pool")
        for column in ("bank_account_assets", "bond_assets", "stock_assets")
    }


def test_production_adult_care_contract_refuses_missing_sstb_before_callback() -> None:
    contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[
        source_producer_name("with_us_adult_care_inputs")
    ]
    sstb_input = next(
        item
        for item in contract.inputs
        if item.column == "sstb_self_employment_income_before_lsr"
        and item.producing_stage
        == transfer_producer_name("person", "puf_tax_itemization__batch_5")
    )
    invoked = False

    def callback() -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)source:with_us_adult_care_inputs.*"
            r"person\.sstb_self_employment_income_before_lsr.*43260 unfilled.*"
            r"asec_source.*transfer:person/puf_tax_itemization__batch_5"
        ),
    ):
        run_producer_when_ready(
            contract,
            callback,
            unfilled_rows={
                item: 43_260 if item == sstb_input else 0 for item in contract.inputs
            },
            invalid_rows={item: 0 for item in contract.inputs},
            absence_receipts={},
        )

    assert invoked is False


def test_canonical_us_late_schedule_is_import_validated_and_byte_stable() -> None:
    reverse_registry = OrderedDict(
        reversed(tuple(CANONICAL_US_LATE_PRODUCER_REGISTRY.items()))
    )
    reconstructed = derive_producer_schedule(
        reverse_registry,
        external_stages=US_LATE_EXTERNAL_STAGES,
    )

    assert reconstructed == CANONICAL_US_LATE_PRODUCER_SCHEDULE
    receipt = us_late_producer_schedule_receipt()
    assert receipt["schema_version"] == 4
    assert receipt["status"] == "derived_and_import_validated"
    assert receipt["schedule_sha256"] == reconstructed.sha256
    assert receipt["producer_count"] == 37
    assert receipt["source_producer_count"] == 16
    assert receipt["transfer_group_count"] == 19
    assert receipt["transfer_target_count"] == 70
    assert receipt["order"][0] == US_LATE_PRIMARY_PUF_STAGE


def test_every_post_clone_source_has_a_nonempty_full_input_inventory() -> None:
    assert set(US_LATE_SOURCE_INPUT_INVENTORIES) == set(
        POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    )
    for operator, inventory in US_LATE_SOURCE_INPUT_INVENTORIES.items():
        assert inventory.operator == operator
        assert inventory.requirements
        assert all(requirement.alternatives for requirement in inventory.requirements)


def test_every_transfer_declares_predictors_and_optional_absence_receipts() -> None:
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[group.name]
        effective_inputs = {
            item.column: item for item in contract.inputs if item.column.startswith("@")
        }
        assert {
            "@effective:age",
            "@effective:is_female",
            "@effective:state_fips",
            "@effective:resolved_person_weight",
            "@effective:resolved_target_weight",
            "@effective:optional_investment_income",
        } <= set(effective_inputs)
        assert effective_inputs[
            "@effective:optional_investment_income"
        ].tolerated_absence_receipts == (
            f"optional_input:{group.name}:optional_investment_income",
        )


def test_production_registry_preserves_finite_numeric_input_kinds() -> None:
    cases = (
        (
            US_LATE_PRIMARY_PUF_STAGE,
            "@effective:employment_income",
            "person",
            "employment_income_before_lsr",
        ),
        (
            source_producer_name("with_us_adult_care_inputs"),
            "@effective:sstb_earned_income",
            "person",
            "sstb_self_employment_income_before_lsr",
        ),
        (
            source_producer_name("with_us_education_inputs"),
            "@effective:qualified_tuition",
            "person",
            "qualified_tuition_expenses",
        ),
        (
            transfer_producer_name("person", "adult_care"),
            "@effective:optional_employment_income",
            "person",
            "employment_income_before_lsr",
        ),
    )

    for producer, input_name, entity, column in cases:
        effective_input = next(
            item
            for item in CANONICAL_US_LATE_PRODUCER_REGISTRY[producer].inputs
            if item.column == input_name
        )
        declared_column = next(
            item
            for alternative in effective_input.alternatives
            for item in alternative
            if (item.entity, item.column) == (entity, column)
        )
        assert declared_column.value_kind == "finite_numeric"


def test_source_contracts_match_strict_runtime_input_semantics() -> None:
    adult = CANONICAL_US_LATE_PRODUCER_REGISTRY[
        source_producer_name("with_us_adult_care_inputs")
    ]
    adult_inputs = {item.column: item for item in adult.inputs}
    assert adult_inputs["@effective:support_role"].alternatives == (
        (
            next(
                column
                for alternative in adult_inputs["@effective:support_role"].alternatives
                for column in alternative
                if column.column == "person_support_channel"
            ),
            next(
                column
                for alternative in adult_inputs["@effective:support_role"].alternatives
                for column in alternative
                if column.column == "person_support_clone_index"
            ),
        ),
    )
    for logical_input, physical_column in (
        ("@effective:raw_person:PEDISDRS", "PEDISDRS"),
        (
            "@effective:raw_person:is_full_time_college_student",
            "is_full_time_college_student",
        ),
    ):
        requirement = adult_inputs[logical_input]
        assert {
            column.value_kind
            for alternative in requirement.alternatives
            for column in alternative
            if column.column == physical_column
        } == {"finite_numeric"}

    education = CANONICAL_US_LATE_PRODUCER_REGISTRY[
        source_producer_name("with_us_education_inputs")
    ]
    education_source = next(
        item
        for item in education.inputs
        if item.column == "@effective:education_source_or_sidecar"
    )
    ed_val = next(
        column
        for alternative in education_source.alternatives
        for column in alternative
        if column.column == "ED_VAL"
    )
    assert ed_val.value_kind == "finite_numeric"


def test_transfer_numeric_predictor_alternatives_are_all_finite() -> None:
    numeric_requirements = {
        "@effective:optional_social_security_income",
        "@effective:optional_retirement_income",
        "@effective:optional_investment_income",
    }
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[group.name]
        inputs = {item.column: item for item in contract.inputs}
        for logical_input in numeric_requirements:
            assert {
                column.value_kind
                for alternative in inputs[logical_input].alternatives
                for column in alternative
            } == {"finite_numeric"}
