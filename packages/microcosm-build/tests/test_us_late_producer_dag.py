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

    assert len(registry) == 37
    assert len(groups) == 19
    assert sum(len(group.targets) for group in groups) == 70
    assert {contract.kind for contract in registry.values()} == {
        "primary_puf",
        "post_clone_source",
        "late_transfer",
        "source_finalizer",
    }
    assert len(registry[US_LATE_PRIMARY_PUF_STAGE].inputs) == 46
    primary_outputs = registry[US_LATE_PRIMARY_PUF_STAGE].outputs
    assert len(primary_outputs) == 100
    assert sum(output.coverage_scope == "puf_clone" for output in primary_outputs) == 65
    assert (
        sum(output.coverage_scope == "whole_pool" for output in primary_outputs) == 35
    )
    assert {
        (output.entity, output.column, output.coverage_scope)
        for output in primary_outputs
        if output.coverage_scope == "whole_pool"
    } >= {
        ("person", "person_support_clone_index", "whole_pool"),
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
    assert receipt["schema_version"] == 6
    assert receipt["execution_receipt_contract"] == {
        "version": 1,
        "row_binding": (
            "declared_input_and_output_content_callback_receipt_and_"
            "previous_execution_sha256"
        ),
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
        "would_claim_wic",
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
