"""Late-stage producer DAG doctrine regressions (microcosm#653)."""

from __future__ import annotations

from collections import Counter, OrderedDict

import pytest

import microcosm.build.us_runtime.acs_pums as acs_pums_module
import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
from microcosm.build.us_runtime.acs_income_universe import (
    ACS_PUMS_EARNINGS_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.education_inputs import (
    US_EDUCATION_INPUTS_OUTPUT_COLUMNS,
    US_EDUCATION_INPUTS_OWNED_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.late_producer_dag import (
    ProducerContract,
    ProducerInput,
    ProducerInputColumn,
    ProducerOutput,
    derive_producer_schedule,
    run_producer_when_ready,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_OPERATOR_CONTRACTS,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
)
from microcosm.build.us_runtime.operator_boundary import (
    FORMULA_OWNED_SOURCE_COLUMNS,
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    US_LATE_OVERLAP_OWNERSHIP_TARGETS,
    US_LATE_SOURCE_CALLBACK_PASSTHROUGH_OUTPUTS,
    us_late_overlap_ownership_receipt,
    validate_us_late_overlap_ownership_receipt,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_PRODUCER_REGISTRY,
    CANONICAL_US_LATE_PRODUCER_SCHEDULE,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    US_LATE_ACS_EARNINGS_UNIVERSE_INPUT_INVENTORY,
    US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
    US_LATE_EXTERNAL_STAGES,
    US_LATE_PRIMARY_PUF_INPUT_INVENTORY,
    US_LATE_PRIMARY_PUF_STAGE,
    US_LATE_SOURCE_FINALIZER_STAGE,
    US_LATE_SOURCE_INPUT_INVENTORIES,
    US_LATE_TRANSFER_INPUT_INVENTORIES,
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


def test_declared_absence_rejects_a_cross_producer_receipt() -> None:
    receipt_id = "optional_input:consumer:predictor"
    requirement = ProducerInput(
        entity="person",
        column="@effective:predictor",
        required_scope="whole_pool",
        producing_stage="post_clone_input_surface",
        tolerated_absence_receipts=(receipt_id,),
    )
    consumer = ProducerContract("consumer", "fixture", (requirement,), ())
    wrong_owner = {
        receipt_id: {
            "receipt_id": receipt_id,
            "status": "declared_absence",
            "entity": "person",
            "column": "@effective:predictor",
            "required_scope": "whole_pool",
            "rows": 1,
            "producer": "different_consumer",
            "reason": "optional availability-pattern input",
        }
    }

    with pytest.raises(
        ValueError,
        match=r"(?s)consumer.*@effective:predictor.*1 unfilled",
    ):
        run_producer_when_ready(
            consumer,
            lambda: pytest.fail("cross-producer absence reached callback"),
            unfilled_rows={requirement: 1},
            invalid_rows={requirement: 0},
            absence_receipts=wrong_owner,
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


def _scoped_dependency_registry(
    *,
    output_scope: str,
    required_scope: str,
) -> dict[str, ProducerContract]:
    return {
        "producer": ProducerContract(
            "producer",
            "fixture",
            (),
            (ProducerOutput("person", "shared", output_scope),),
        ),
        "consumer": ProducerContract(
            "consumer",
            "fixture",
            (
                ProducerInput(
                    "person",
                    "shared",
                    required_scope,
                    "producer",
                ),
            ),
            (),
        ),
    }


def test_schedule_rejects_producer_output_with_insufficient_scope() -> None:
    registry = _scoped_dependency_registry(
        output_scope="asec_source",
        required_scope="whole_pool",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)scope_mismatches=.*consumer.*producer.*person\.shared.*"
            r"whole_pool.*asec_source"
        ),
    ):
        derive_producer_schedule(registry)


@pytest.mark.parametrize(
    ("output_scope", "required_scope"),
    (
        ("whole_pool", "whole_pool"),
        ("whole_pool", "asec_source"),
        ("whole_pool", "puf_clone"),
        ("asec_source", "asec_source"),
        ("puf_clone", "puf_clone"),
        ("receipt", "whole_pool"),
    ),
)
def test_schedule_accepts_declared_scope_coverage(
    output_scope: str,
    required_scope: str,
) -> None:
    schedule = derive_producer_schedule(
        _scoped_dependency_registry(
            output_scope=output_scope,
            required_scope=required_scope,
        )
    )

    assert schedule.edges == (("producer", "consumer"),)
    assert schedule.waves == (("producer",), ("consumer",))


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

    assert len(registry) == 38
    assert len(groups) == 19
    assert sum(len(group.targets) for group in groups) == 70
    assert {contract.kind for contract in registry.values()} == {
        "primary_puf",
        "acs_earnings_universe",
        "post_clone_source",
        "late_transfer",
        "source_finalizer",
    }
    assert len(registry[US_LATE_PRIMARY_PUF_STAGE].inputs) == 119
    primary_outputs = registry[US_LATE_PRIMARY_PUF_STAGE].outputs
    assert len(primary_outputs) == 100
    assert sum(output.coverage_scope == "puf_clone" for output in primary_outputs) == 64
    assert (
        sum(output.coverage_scope == "whole_pool" for output in primary_outputs) == 35
    )
    assert sum(output.coverage_scope == "acs_source" for output in primary_outputs) == 1
    assert {
        (output.entity, output.column, output.coverage_scope)
        for output in primary_outputs
        if output.coverage_scope == "acs_source"
    } == {
        ("household", "TYPEHUGQ", "acs_source"),
    }
    assert {
        (output.entity, output.column, output.coverage_scope)
        for output in primary_outputs
        if output.coverage_scope == "whole_pool"
    } >= {
        ("person", "person_support_clone_index", "whole_pool"),
        ("person", "s_corp_income", "whole_pool"),
        ("frame", "@us_puf_clone_attachment_manifest", "whole_pool"),
    }
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


def test_late_overlap_ownership_exhausts_every_permitted_dual_write() -> None:
    primary = {
        (entity, column)
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            "primary_puf_qrf"
        ].items()
        for column in columns
    }
    source_writes = {
        (entity, column)
        for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            POOL_OPERATOR_CONTRACTS[operator].family
        ].items()
        for column in columns
        if column not in FORMULA_OWNED_SOURCE_COLUMNS.get(entity, ())
    }
    callback_passthroughs = {
        target
        for targets in US_LATE_SOURCE_CALLBACK_PASSTHROUGH_OUTPUTS.values()
        for target in targets
    }
    source_touches = source_writes | callback_passthroughs
    transfer = {
        (group.entity, target)
        for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        for target in group.targets
    }
    tail_owned = {
        *(("person", column) for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS),
        *(("tax_unit", column) for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS),
    }
    recipient_owned = {
        *(("person", column) for column in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS),
        *(("tax_unit", column) for column in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS),
    } - tail_owned
    declared = set(US_LATE_OVERLAP_OWNERSHIP_TARGETS)

    assert len(primary) == 65
    assert len(source_writes) == 35
    assert len(source_writes & transfer) == 29
    assert len(transfer) == 70
    assert len(recipient_owned) == 60
    assert (
        callback_passthroughs
        == {
            ("person", column)
            for column in set(US_EDUCATION_INPUTS_OUTPUT_COLUMNS)
            - set(US_EDUCATION_INPUTS_OWNED_OUTPUT_COLUMNS)
        }
        == {("person", "qualified_tuition_expenses")}
    )
    assert primary & source_touches & transfer & recipient_owned == declared
    assert primary & source_writes & transfer & recipient_owned == {
        ("person", "traditional_ira_contributions_desired"),
        ("person", "self_employed_pension_contributions_desired"),
    }
    assert primary & source_touches & transfer & tail_owned == set()

    receipt = dict(us_late_overlap_ownership_receipt())
    assert validate_us_late_overlap_ownership_receipt(receipt) == receipt["sha256"]
    assert len(receipt["targets"]) == 3
    assert len(receipt["ownership"]) == 18
    assert {(row["entity"], row["target"]) for row in receipt["ownership"]} == declared
    assert {(row["origin"], row["clone_index"]) for row in receipt["ownership"]} == {
        (origin, clone) for origin in ("asec", "acs") for clone in range(3)
    }
    for row in receipt["ownership"]:
        assert sum(action["owns_final"] for action in row["producer_actions"]) == 1
        owner = next(
            action["producer"]
            for action in row["producer_actions"]
            if action["owns_final"]
        )
        assert owner == row["final_owner"]

    source_action_by_cell = {
        (row["target"], row["origin"], row["clone_index"]): next(
            action["action"]
            for action in row["producer_actions"]
            if action["producer"] == "source:with_us_education_inputs"
        )
        for row in receipt["ownership"]
        if row["target"] == "qualified_tuition_expenses"
    }
    for clone_index in range(3):
        assert (
            source_action_by_cell[("qualified_tuition_expenses", "asec", clone_index)]
            == "consume_only_byte_exact_noop"
        )
        assert (
            source_action_by_cell[("qualified_tuition_expenses", "acs", clone_index)]
            == "origin_projection_masked_noop"
        )

    owner_by_cell = {
        (row["target"], row["origin"], row["clone_index"]): row["final_owner"]
        for row in receipt["ownership"]
    }
    for origin in ("asec", "acs"):
        assert owner_by_cell[("qualified_tuition_expenses", origin, 0)] == (
            "transfer:person/puf_tax_itemization__batch_2"
        )
        for clone_index in (1, 2):
            assert (
                owner_by_cell[("qualified_tuition_expenses", origin, clone_index)]
                == US_LATE_PRIMARY_PUF_STAGE
            )
    for target in (
        "traditional_ira_contributions_desired",
        "self_employed_pension_contributions_desired",
    ):
        for clone_index in range(3):
            assert owner_by_cell[(target, "asec", clone_index)] == source_producer_name(
                "with_us_retirement_contribution_inputs"
            )
        assert owner_by_cell[(target, "acs", 0)] == transfer_producer_name(
            "person",
            "puf_tax_itemization__batch_2"
            if target == "traditional_ira_contributions_desired"
            else "puf_tax_itemization__batch_3",
        )
        for clone_index in (1, 2):
            assert owner_by_cell[(target, "acs", clone_index)] == (
                US_LATE_PRIMARY_PUF_STAGE
            )

    finalization_by_cell = {
        (row["target"], row["origin"], row["clone_index"]): row["finalization"]
        for row in receipt["ownership"]
    }
    for target in (
        "traditional_ira_contributions_desired",
        "self_employed_pension_contributions_desired",
    ):
        assert finalization_by_cell[(target, "asec", 2)] == (
            "byte_exact_clone_1_mirror"
        )

    schedule_receipt = us_late_producer_schedule_receipt()
    assert schedule_receipt["overlap_ownership"] == receipt


def test_primary_puf_inventory_declares_exact_read_before_write_surface() -> None:
    requirements = {
        requirement.label: requirement
        for requirement in US_LATE_PRIMARY_PUF_INPUT_INVENTORY.requirements
    }

    assert len(requirements) == 114
    assert tuple(
        (item.entity, item.column, item.value_kind)
        for item in requirements["filing_status"].alternatives[0]
    ) == (("tax_unit", "filing_status_input", "non_null"),)
    assert requirements["age"].alternatives[0][0].value_kind == "finite_numeric"
    allocation_basis = {
        label.removeprefix("person_output_allocation_basis:")
        for label in requirements
        if label.startswith("person_output_allocation_basis:")
    }
    assert allocation_basis == set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS)
    assert all(
        requirements[f"person_output_allocation_basis:{column}"].optional
        for column in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    )
    tax_unit_passthrough = {
        label.removeprefix("tax_unit_output_passthrough:")
        for label in requirements
        if label.startswith("tax_unit_output_passthrough:")
    }
    assert tax_unit_passthrough == set(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS)
    assert all(
        requirements[f"tax_unit_output_passthrough:{column}"].optional
        for column in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    )
    assert requirements["qualified_tuition_allocation_fallback"].optional

    primary = CANONICAL_US_LATE_PRODUCER_REGISTRY[US_LATE_PRIMARY_PUF_STAGE]
    contract_inputs = {item.column: item for item in primary.inputs}
    for column in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS:
        label = f"person_output_allocation_basis:{column}"
        requirement = requirements[label]
        assert tuple(
            (item.entity, item.column, item.value_kind)
            for item in requirement.alternatives[0]
        ) == (("person", column, "finite_numeric"),)
        assert contract_inputs[f"@effective:{label}"].tolerated_absence_receipts == (
            f"optional_input:{US_LATE_PRIMARY_PUF_STAGE}:{label}",
        )
    for column in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS:
        label = f"tax_unit_output_passthrough:{column}"
        requirement = requirements[label]
        assert tuple(
            (item.entity, item.column, item.value_kind)
            for item in requirement.alternatives[0]
        ) == (("tax_unit", column, "finite_numeric"),)
        assert contract_inputs[f"@effective:{label}"].tolerated_absence_receipts == (
            f"optional_input:{US_LATE_PRIMARY_PUF_STAGE}:{label}",
        )
    fallback = requirements["qualified_tuition_allocation_fallback"]
    assert tuple(
        (item.entity, item.column, item.value_kind) for item in fallback.alternatives[0]
    ) == (("person", "is_full_time_college_student", "finite_numeric"),)
    assert contract_inputs[
        "@effective:qualified_tuition_allocation_fallback"
    ].tolerated_absence_receipts == (
        f"optional_input:{US_LATE_PRIMARY_PUF_STAGE}:"
        "qualified_tuition_allocation_fallback",
    )
    typehugq = requirements["validated_structure:TYPEHUGQ"]
    assert typehugq.required_scope == "acs_source"
    declared_typehugq = contract_inputs["@effective:validated_structure:TYPEHUGQ"]
    assert declared_typehugq.required_scope == "acs_source"
    assert declared_typehugq.tolerated_absence_receipts == ()
    raw_inputs = {
        item.column: item
        for item in primary.inputs
        if item.column in set(ACS_PUMS_EARNINGS_SOURCE_COLUMNS.values())
    }
    assert set(raw_inputs) == set(ACS_PUMS_EARNINGS_SOURCE_COLUMNS.values())
    for declared in raw_inputs.values():
        assert declared.required_scope == "acs_source"
        assert declared.producing_stage == US_LATE_EXTERNAL_STAGES[0]
        assert declared.tolerated_absence_receipts == ()
        assert declared.alternatives[0][0].value_kind == "column_present"


def test_canonical_us_late_registry_declares_required_cross_producer_edges() -> None:
    edges = set(CANONICAL_US_LATE_PRODUCER_SCHEDULE.edges)

    assert len(edges) == 71
    assert (
        US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
        US_LATE_PRIMARY_PUF_STAGE,
    ) in edges
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
    assert CANONICAL_US_LATE_PRODUCER_SCHEDULE.waves[:2] == (
        (US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,),
        (US_LATE_PRIMARY_PUF_STAGE,),
    )
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
    assert any(
        item.column == "@source_finalizer_execution_config"
        and item.producing_stage == US_LATE_EXTERNAL_STAGES[0]
        for item in CANONICAL_US_LATE_PRODUCER_REGISTRY[
            US_LATE_SOURCE_FINALIZER_STAGE
        ].inputs
    )
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


def test_adult_care_transfer_declares_role_and_refuses_before_callback() -> None:
    contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[
        transfer_producer_name("person", "adult_care")
    ]
    role_input = next(
        item
        for item in contract.inputs
        if item.column == "@effective:adult_care_tax_unit_role"
    )
    assert role_input.alternatives == (
        (ProducerInputColumn("person", "tax_unit_role_input"),),
    )
    assert role_input.required_scope == "whole_pool"
    assert role_input.producing_stage == US_LATE_EXTERNAL_STAGES[0]
    invoked = False

    def callback() -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)transfer:person/adult_care.*"
            r"person\.@effective:adult_care_tax_unit_role.*1 unfilled.*"
            r"whole_pool.*post_clone_input_surface"
        ),
    ):
        run_producer_when_ready(
            contract,
            callback,
            unfilled_rows={
                item: 1 if item == role_input else 0 for item in contract.inputs
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
    assert receipt["schema_version"] == 16
    assert receipt["execution_receipt_contract"] == {
        "version": 3,
        "row_binding": (
            "declared_globally_reconciled_input_and_scope_exact_output_source_"
            "and_primary_callback_resource_receipt_and_previous_execution_sha256"
        ),
        "virtual_resource_binding": ("exact_kind_specific_semantic_payload_and_sha256"),
        "top_binding": (
            "entry_and_output_frame_sha256_execution_chain_source_"
            "completion_and_nineteen_transfer_groups"
        ),
        "transition_authority": {
            "authority_id": "us_stacked_late_producer_transition",
            "metadata_key": "us_late_producer_transition_authority",
            "version": 1,
            "independent_digest_required": True,
        },
    }
    assert receipt["status"] == "derived_and_import_validated"
    assert receipt["schedule_sha256"] == reconstructed.sha256
    assert receipt["producer_count"] == 38
    assert receipt["source_producer_count"] == 16
    assert receipt["transfer_group_count"] == 19
    assert receipt["transfer_target_count"] == 70
    assert receipt["order"][:2] == [
        US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
        US_LATE_PRIMARY_PUF_STAGE,
    ]


def test_every_post_clone_source_has_a_nonempty_full_input_inventory() -> None:
    assert set(US_LATE_SOURCE_INPUT_INVENTORIES) == set(
        POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    )
    for operator, inventory in US_LATE_SOURCE_INPUT_INVENTORIES.items():
        assert inventory.operator == operator
        assert inventory.requirements
        assert all(requirement.alternatives for requirement in inventory.requirements)
        physical_columns = {
            column.column
            for requirement in inventory.requirements
            for alternative in requirement.alternatives
            for column in alternative
        }
        assert "@post_clone_source_execution_config" in physical_columns
        assert "@weeks_unemployed_sidecar" not in physical_columns
        assert "@education_assistance_sidecar" not in physical_columns


def test_acs_earnings_universe_declares_every_receipt_affecting_input() -> None:
    inventory = US_LATE_ACS_EARNINGS_UNIVERSE_INPUT_INVENTORY

    assert len(inventory.requirements) == 10
    assert {requirement.label for requirement in inventory.requirements} == {
        "age",
        "support_channel",
        "person_tax_unit_link",
        "support_clone_index",
        "stable_person_lineage",
        "raw_source:WAGP",
        "raw_source:SEMP",
        "mapped_earnings:employment_income_before_lsr",
        "mapped_earnings:self_employment_income_before_lsr",
        "execution_config",
    }
    by_label = {
        requirement.label: requirement for requirement in inventory.requirements
    }
    assert not by_label["raw_source:WAGP"].optional
    assert not by_label["raw_source:SEMP"].optional
    assert {
        by_label[label].alternatives[0][0].value_kind
        for label in ("raw_source:WAGP", "raw_source:SEMP")
    } == {"column_present"}
    assert by_label["mapped_earnings:employment_income_before_lsr"].optional
    assert by_label["mapped_earnings:self_employment_income_before_lsr"].optional
    lineage = next(
        requirement
        for requirement in inventory.requirements
        if requirement.label == "stable_person_lineage"
    )
    assert {
        tuple((column.entity, column.column) for column in alternative)
        for alternative in lineage.alternatives
    } == {
        (("person", "person_source_id"),),
        (("person", "person_id"),),
    }


def test_every_transfer_declares_predictors_and_optional_absence_receipts() -> None:
    primary_inputs = {
        item.column
        for item in CANONICAL_US_LATE_PRODUCER_REGISTRY[
            US_LATE_PRIMARY_PUF_STAGE
        ].inputs
    }
    assert {
        "@effective:puf_donor",
        "@effective:primary_qrf_bank",
        "@effective:primary_puf_execution_config",
    } <= primary_inputs

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
            "@effective:late_transfer_model_config",
            "@effective:late_transfer_target_bank",
        } <= set(effective_inputs)
        assert effective_inputs[
            "@effective:optional_investment_income"
        ].tolerated_absence_receipts == (
            f"optional_input:{group.name}:optional_investment_income",
        )


def test_every_transfer_declares_complete_cross_grain_validation_surface() -> None:
    entities = (
        "person",
        "household",
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    )
    groups = entities[1:]
    expected = {
        *((entity, f"{entity}_support_channel") for entity in entities),
        *((entity, f"{entity}_support_clone_index") for entity in entities),
        *(("person", f"person_{entity}_id") for entity in groups),
        *((entity, f"{entity}_id") for entity in groups),
        ("person", "person_id"),
        ("person", "person_spine_source_id"),
        ("person", "person_source_id"),
        ("household", "household_spine_source_id"),
        ("household", "household_source_id"),
        ("household", "TYPEHUGQ"),
        ("household", "@resolved_weight"),
        ("frame", "@us_spine_assembly_manifest"),
        ("frame", "@us_stacked_spine_manifest"),
        ("frame", "@us_puf_clone_attachment_manifest"),
    }
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        inventory = US_LATE_TRANSFER_INPUT_INVENTORIES[group.name]
        physical = {
            (column.entity, column.column)
            for requirement in inventory.requirements
            for alternative in requirement.alternatives
            for column in alternative
        }
        assert expected <= physical

        clone_columns = {
            (column.entity, column.column, column.value_kind)
            for requirement in inventory.requirements
            for alternative in requirement.alternatives
            for column in alternative
            if column.column.endswith("_support_clone_index")
        }
        assert clone_columns == {
            (entity, f"{entity}_support_clone_index", "finite_numeric")
            for entity in entities
        }


def test_every_origin_exclusive_raw_input_has_its_native_scope() -> None:
    acs_raw_inputs = {
        *(("household", column) for column in acs_pums_module._HOUSEHOLD_FRAME_COLUMNS),
        *(("person", column) for column in acs_pums_module._PERSON_REQUIRED),
    }
    asec_person_raw_columns = {
        "A_HSCOL",
        "A_MJOCC",
        "CAID",
        "CHAMPVA",
        "CHSP_VAL",
        "CSP_VAL",
        "DIS_SC1",
        "DIS_SC2",
        "DIS_VAL1",
        "DIS_VAL2",
        "DST_SC1",
        "DST_SC2",
        "DST_SC3",
        "DST_SC4",
        "DST_VAL1",
        "DST_VAL2",
        "DST_VAL3",
        "DST_VAL4",
        "ED_VAL",
        "IHSFLG",
        "I_ERNVAL",
        "I_SEVAL",
        "LKWEEKS",
        "MCARE",
        "MIL",
        "PEAFEVER",
        "PEDISDRS",
        "PEINUSYR",
        "PEIO1COW",
        "PENATVTY",
        "PEN_SC1",
        "PEN_SC2",
        "PERIDNUM",
        "PRCITSHP",
        "RESNSS1",
        "RESNSS2",
        "RETCB_VAL",
        "SPM_CAPHOUSESUB",
        "SPM_CHILDCAREXPNS",
        "SPM_ENGVAL",
        "SSI_YN",
        "SS_YN",
        "UC_VAL",
        "WC_VAL",
    }
    required_scope = {
        **{key: "acs_source" for key in acs_raw_inputs},
        **{("person", column): "asec_source" for column in asec_person_raw_columns},
        ("household", "H_TENURE"): "asec_source",
    }
    expected_counts = {
        ("household", "TYPEHUGQ"): 39,
        ("person", "MCARE"): 2,
        ("person", "PERIDNUM"): 18,
        ("person", "SEMP"): 2,
        ("person", "WAGP"): 2,
        **{
            ("person", column): 1
            for column in asec_person_raw_columns
            if column
            not in {
                "DST_SC3",
                "DST_SC4",
                "DST_VAL3",
                "DST_VAL4",
                "MCARE",
                "PERIDNUM",
            }
        },
    }

    observed: Counter[tuple[str, str, str]] = Counter()
    receipts: dict[tuple[str, str], set[str]] = {}
    for contract in CANONICAL_US_LATE_PRODUCER_REGISTRY.values():
        for requirement in contract.inputs:
            for alternative in requirement.alternatives:
                for column in alternative:
                    key = (column.entity, column.column)
                    if key not in required_scope:
                        continue
                    observed[
                        (column.entity, column.column, requirement.required_scope)
                    ] += 1
                    receipts.setdefault(key, set()).update(
                        requirement.tolerated_absence_receipts
                    )

    assert observed == Counter(
        {(*key, required_scope[key]): count for key, count in expected_counts.items()}
    )
    assert sum(observed.values()) == 101
    assert sum(count for key, count in observed.items() if key[2] == "acs_source") == 43
    assert (
        sum(count for key, count in observed.items() if key[2] == "asec_source") == 58
    )
    assert receipts[("household", "TYPEHUGQ")] == set()

    execution_identity = acs_transfer_module.acs_transfer_execution_contract_identity()
    assert execution_identity["schema_version"] == 2
    assert execution_identity["housing"]["head_source_precedence"] == [
        {"source": "is_household_head", "head_codes": [True]},
        {"source": "A_EXPRRP", "head_codes": [1, 2]},
        {"source": "A_LINENO", "head_codes": [1]},
    ]
    assert execution_identity["housing"]["tenure_source_precedence"] == [
        "tenure_type",
        "spm_unit_tenure_type",
    ]


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
        if item.column == "@effective:education_source"
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


def test_source_numeric_input_audit_is_fully_executable() -> None:
    expected_finite = {
        "with_us_prior_year_income_inputs": {
            "source_year",
            "WSAL_VAL",
            "SEMP_VAL",
            "I_ERNVAL",
            "I_SEVAL",
            "employment_income_last_year",
            "self_employment_income_last_year",
        },
        "with_us_medicare_take_up_input": {"MCARE"},
        "with_us_pregnancy_inputs": {"A_SEX", "A_AGE"},
        "with_us_wic_claim_input": {
            "age",
            "is_female",
            "is_pregnant",
            "own_children_in_household",
            "person_family_id",
        },
        "impute_us_housing_assistance_to_puf_support": {
            "receives_housing_assistance",
            "takes_up_housing_assistance_if_eligible",
        },
        "with_us_child_support_inputs": {"CSP_VAL", "CHSP_VAL"},
        "with_us_disability_benefits": {
            "DIS_VAL1",
            "DIS_SC1",
            "DIS_VAL2",
            "DIS_SC2",
        },
        "with_us_workers_compensation": {"WC_VAL"},
        "with_us_weeks_unemployed": {
            "source_year",
            "PERIDNUM",
            "LKWEEKS",
            "age",
            "A_AGE",
            "is_male",
            "is_female",
            "A_SEX",
            "tax_unit_is_joint",
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
            "unemployment_compensation",
            "UC_VAL",
        },
        "with_us_childcare_inputs": {"person_spm_unit_id", "SPM_CHILDCAREXPNS"},
        "with_us_adult_care_inputs": {
            "PEDISDRS",
            "is_full_time_college_student",
            "person_id",
            "person_support_clone_index",
        },
        "with_us_energy_subsidy_input": {"person_spm_unit_id", "SPM_ENGVAL"},
        "with_us_retirement_contribution_inputs": {
            "RETCB_VAL",
            "WSAL_VAL",
            "SEMP_VAL",
        },
        "with_us_retirement_distribution_inputs": {
            "DST_SC1",
            "DST_VAL1",
            "DST_SC2",
            "DST_VAL2",
            "DST_SC1_YNG",
            "DST_VAL1_YNG",
            "DST_SC2_YNG",
            "DST_VAL2_YNG",
            "taxable_ira_distributions",
        },
        "with_us_immigration_inputs": {
            "PRCITSHP",
            "PEINUSYR",
            "PENATVTY",
            "A_AGE",
            "A_MARITL",
            "A_SPOUSE",
            "A_HSCOL",
            "WSAL_VAL",
            "SEMP_VAL",
            "MCARE",
            "CAID",
            "IHSFLG",
            "CHAMPVA",
            "MIL",
            "PEN_SC1",
            "PEN_SC2",
            "RESNSS1",
            "RESNSS2",
            "SS_YN",
            "SSI_YN",
            "PEIO1COW",
            "A_MJOCC",
            "PEAFEVER",
            "SPM_CAPHOUSESUB",
        },
        "with_us_education_inputs": {"ED_VAL", "qualified_tuition_expenses"},
    }
    assert set(expected_finite) == set(US_LATE_SOURCE_INPUT_INVENTORIES)
    for operator, expected_columns in expected_finite.items():
        inventory = US_LATE_SOURCE_INPUT_INVENTORIES[operator]
        finite_columns = {
            column.column
            for requirement in inventory.requirements
            for alternative in requirement.alternatives
            for column in alternative
            if column.value_kind == "finite_numeric"
        }
        assert expected_columns <= finite_columns, operator

    common_role_operators = {
        "with_us_prior_year_income_inputs",
        "impute_us_housing_assistance_to_puf_support",
        "with_us_child_support_inputs",
        "with_us_disability_benefits",
        "with_us_workers_compensation",
        "with_us_childcare_inputs",
        "with_us_energy_subsidy_input",
        "with_us_retirement_contribution_inputs",
        "with_us_retirement_distribution_inputs",
    }
    for operator in common_role_operators:
        sex = next(
            requirement
            for requirement in US_LATE_SOURCE_INPUT_INVENTORIES[operator].requirements
            if requirement.label == "sex"
        )
        assert {
            column.value_kind
            for alternative in sex.alternatives
            for column in alternative
        } == {"finite_numeric"}


def test_late_target_dependency_kinds_partition_51_numeric_17_boolean_2_string() -> (
    None
):
    string_targets = {"ssn_card_type", "immigration_status_str"}
    boolean_targets = {
        "is_incapable_of_self_care",
        "is_pregnant",
        "estate_income_would_be_qualified",
        "farm_operations_income_would_be_qualified",
        "farm_rent_income_would_be_qualified",
        "partnership_s_corp_income_would_be_qualified",
        "rental_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
        "sstb_self_employment_income_would_be_qualified",
        "business_is_sstb",
        "attends_eligible_educational_institution_for_american_opportunity_credit",
        "has_american_opportunity_credit_1098_t_or_exception",
        "has_american_opportunity_credit_institution_ein",
        "is_enrolled_at_least_half_time_for_american_opportunity_credit",
        "is_pursuing_credential_for_american_opportunity_credit",
        "takes_up_medicare_if_eligible",
        "takes_up_wic_if_eligible",
    }
    observed: dict[str, set[str]] = {}
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[group.name]
        for target in group.targets:
            direct_inputs = [item for item in contract.inputs if item.column == target]
            assert direct_inputs
            observed[target] = {
                column.value_kind
                for item in direct_inputs
                for alternative in item.alternatives
                for column in alternative
            }
    numeric_targets = set(observed) - boolean_targets - string_targets
    assert (len(numeric_targets), len(boolean_targets), len(string_targets)) == (
        51,
        17,
        2,
    )
    assert all(observed[target] == {"finite_numeric"} for target in numeric_targets)
    assert all(
        observed[target] == {"non_null"} for target in boolean_targets | string_targets
    )
