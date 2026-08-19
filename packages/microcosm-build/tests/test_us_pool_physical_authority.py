"""Bundle-only materialization of the US pool's physical authorities."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from inspect import signature

import pytest

from microcosm.build.frame_sampling import EXACT_COUNT_RULE
from microcosm.build.spec_engine import (
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.spec_engine.battery_semantics import (
    project_battery_legacy_contract,
)
from microcosm.build.spec_engine.canonical import canonical_json_bytes
from microcosm.build.spec_engine.model import FrozenMap, thaw_json
from microcosm.build.us_runtime.acs_transfer import (
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_DERIVE_OPERATOR_ORDER,
    POOL_HOUSEHOLD_MASS_SHARES,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_RANDOM_SEED,
    POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
    POOL_TIME_PERIOD,
    pool_remaining_stage_input_manifest,
    pool_remaining_stage_input_manifest_receipt,
)
from microcosm.build.us_runtime.pool_physical_authority import (
    USPoolPhysicalAuthority,
    USPoolPhysicalAuthorityError,
    compile_us_pool_physical_authority,
)
from microcosm.build.us_runtime.pool_runtime_plan import (
    USPoolRuntimePlan,
    _plan_sha256,
)
from microcosm.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
    PRIMARY_QRF_MANIFEST_FILENAME,
    PRIMARY_QRF_TARGET_ORDER,
    PRIMARY_QRF_TARGET_ORDER_SHA256,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.build.us_runtime.spec_authority import (
    compile_us_spec_authority,
)
from microcosm.build.us_runtime.stacked_battery_contract import (
    build_live_stacked_battery_contract,
)
from microcosm.build.us_runtime.stacked_spine import (
    CANONICAL_STACKED_GAP_FILL_SURFACE,
    CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE,
    CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE,
    CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE,
    StackedAssemblyAuthority,
    StackedGapFillAuthority,
    StackedLateProducerAuthority,
    StackedPrimaryQrfAuthority,
    gap_fill_stacked_spine,
    materialize_stacked_terminal_authority,
    run_stacked_puf_pass,
    stacked_gap_fill_plan,
    stacked_gap_fill_producer_schedule_receipt,
    stacked_spine_authority_receipt,
)
from microcosm.build.us_runtime.take_up_contract import take_up_contract_identity
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_PRODUCER_REGISTRY,
    CANONICAL_US_LATE_PRODUCER_SCHEDULE,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    us_late_producer_schedule_receipt,
)


@pytest.fixture(scope="module")
def plan() -> USPoolRuntimePlan:
    return USPoolRuntimePlan.from_spec_authority(
        compile_us_spec_authority(
            compile_runtime_authorities(compile_spec(load_bundle("us")))
        )
    )


@pytest.fixture(scope="module")
def authority(plan: USPoolRuntimePlan) -> USPoolPhysicalAuthority:
    return compile_us_pool_physical_authority(plan)


def _json_ready(value: object) -> object:
    if isinstance(value, FrozenMap):
        return thaw_json(value)
    if is_dataclass(value):
        return {
            item.name: _json_ready(getattr(value, item.name))
            for item in fields(value)
            if item.name not in {"contract", "node", "transfer_execution_contract"}
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _replace_map(value: FrozenMap, **changes: object) -> FrozenMap:
    entries = dict(value.entries)
    entries.update(changes)
    return FrozenMap(tuple(sorted(entries.items())))


def test_materialized_early_and_late_transfer_authorities_are_exact_oracles(
    authority: USPoolPhysicalAuthority,
) -> None:
    expected_gap_plan = _json_ready(stacked_gap_fill_plan())
    observed_gap_plan = thaw_json(authority.gap_fill.plan_wire())
    assert observed_gap_plan == expected_gap_plan
    assert canonical_json_bytes(observed_gap_plan) == canonical_json_bytes(
        expected_gap_plan
    )
    assert thaw_json(authority.gap_fill.schedule_receipt) == _json_ready(
        stacked_gap_fill_producer_schedule_receipt()
    )

    expected_groups = tuple(
        {
            "name": group.name,
            "entity": group.entity,
            "family": group.family,
            "targets": list(group.targets),
        }
        for group in CANONICAL_US_LATE_TRANSFER_GROUPS
    )
    assert thaw_json(authority.late_producers.transfer_group_wire()) == list(
        expected_groups
    )
    for observed, expected in zip(
        authority.late_producers.transfer_groups,
        CANONICAL_US_LATE_TRANSFER_GROUPS,
        strict=True,
    ):
        assert (
            observed.name,
            observed.entity,
            observed.family,
            observed.targets,
        ) == (expected.name, expected.entity, expected.family, expected.targets)
        assert thaw_json(observed.target_families) == _json_ready(
            expected.target_families
        )
    assert thaw_json(authority.late_producers.schedule_receipt) == _json_ready(
        us_late_producer_schedule_receipt()
    )
    assert dict(authority.late_producers.registry) == dict(
        CANONICAL_US_LATE_PRODUCER_REGISTRY
    )
    assert authority.late_producers.schedule == CANONICAL_US_LATE_PRODUCER_SCHEDULE


def test_primary_qrf_parameters_and_seed_defaults_match_live_kernels(
    authority: USPoolPhysicalAuthority,
) -> None:
    assert authority.primary_qrf.predictors == PUF_TAX_DETAIL_DEFAULT_PREDICTORS
    assert authority.primary_qrf.person_outputs == PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    assert (
        authority.primary_qrf.tax_unit_outputs
        == PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    )
    assert authority.primary_qrf.target_order == PRIMARY_QRF_TARGET_ORDER
    assert authority.primary_qrf.target_order_sha256 == PRIMARY_QRF_TARGET_ORDER_SHA256
    assert (
        authority.primary_qrf.checkpoint_schema_version
        == PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION
    )
    assert authority.primary_qrf.manifest_filename == PRIMARY_QRF_MANIFEST_FILENAME
    assert authority.model.model_seed == POOL_RANDOM_SEED
    assert authority.seeds.model_seed == POOL_RANDOM_SEED
    assert authority.seeds.sampling_seed == 578
    assert authority.model.primary_n_estimators == signature(
        run_stacked_puf_pass
    ).parameters["n_estimators"].default
    assert authority.model.transfer_n_estimators == signature(
        gap_fill_stacked_spine
    ).parameters["n_estimators"].default
    assert (
        authority.model.max_targets_per_fit
        == DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
    )
    assert {site.contract["value_source"] for site in authority.seeds.model_sites} == {
        "run_request.build_model_seed"
    }
    assert {
        site.contract["value_source"] for site in authority.seeds.sampling_sites
    } == {"run_request.sample_seed"}


def test_take_up_and_remaining_stage_materializers_are_byte_exact(
    authority: USPoolPhysicalAuthority,
) -> None:
    expected_identity = take_up_contract_identity()
    assert thaw_json(authority.take_up.identity) == expected_identity
    assert canonical_json_bytes(authority.take_up.identity) == canonical_json_bytes(
        expected_identity
    )
    assert len(authority.take_up.programs) == len(expected_identity["programs"])
    assert authority.take_up.steps
    assert len(authority.take_up.steps) == sum(
        len(program.steps) for program in authority.take_up.programs
    )

    expected_manifest = pool_remaining_stage_input_manifest(
        take_up_program_bindings=authority.take_up.program_bindings
    )
    assert tuple(
        (
            row.stage,
            row.consumer,
            row.entity,
            row.variable,
            row.execution_scope,
            row.provision,
            row.available_by,
            row.fallback,
        )
        for row in authority.remaining_stage.inputs
    ) == tuple(
        (
            row.stage,
            row.consumer,
            row.entity,
            row.variable,
            row.execution_scope,
            row.provision,
            row.available_by,
            row.fallback,
        )
        for row in expected_manifest
    )
    expected_receipt = pool_remaining_stage_input_manifest_receipt(
        take_up_program_bindings=authority.take_up.program_bindings
    )
    assert thaw_json(authority.remaining_stage.checkpoint_identity_receipt) == (
        expected_receipt
    )
    assert (
        authority.remaining_stage.pre_clone_source_operator_order
        == POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert (
        authority.remaining_stage.post_clone_source_operator_order
        == POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert authority.remaining_stage.derive_operator_order == POOL_DERIVE_OPERATOR_ORDER


def test_battery_and_simulation_settings_match_constants_mode(
    authority: USPoolPhysicalAuthority,
) -> None:
    expected_battery = project_battery_legacy_contract(
        build_live_stacked_battery_contract(),
        authority_receipt=stacked_spine_authority_receipt(),
    )
    assert thaw_json(authority.battery.runtime_contract) == expected_battery
    assert canonical_json_bytes(authority.battery.runtime_contract) == (
        canonical_json_bytes(expected_battery)
    )
    assert tuple(gate.kernel for gate in authority.battery.gates) == (
        "kernel:stacked_completeness_gate",
        "kernel:by_origin_battery",
    )
    assert authority.simulation.target_period == POOL_TIME_PERIOD
    assert authority.simulation.model_seed == POOL_RANDOM_SEED
    assert (
        authority.simulation.household_batch_size
        == POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE
    )
    assert authority.simulation.exact_count_rule == EXACT_COUNT_RULE
    assert thaw_json(authority.simulation.household_mass_shares) == dict(
        POOL_HOUSEHOLD_MASS_SHARES
    )
    assert authority.simulation.clone_attachment_fraction_default == 1.0
    assert authority.simulation.sampling_channels == authority.seeds.sampling_channels
    assert thaw_json(
        authority.gap_fill.directions[0].target_families
    ) != _json_ready(CANONICAL_STACKED_GAP_FILL_SURFACE)
    observed_gap_surface: dict[str, dict[str, list[str]]] = {}
    for direction in authority.gap_fill.directions:
        for entity, families in thaw_json(direction.target_families).items():
            observed_gap_surface.setdefault(entity, {}).update(families)
    assert observed_gap_surface == _json_ready(CANONICAL_STACKED_GAP_FILL_SURFACE)
    assert thaw_json(authority.terminal.post_puf_transfer_surface) == _json_ready(
        CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
    )


def test_materialized_values_are_accepted_by_generic_runtime_seams(
    authority: USPoolPhysicalAuthority,
) -> None:
    assembly = StackedAssemblyAuthority(**authority.assembly.materializer_kwargs())
    assert dict(assembly.household_mass_shares) == dict(POOL_HOUSEHOLD_MASS_SHARES)
    assert assembly.mass_anchor_channel == authority.simulation.mass_anchor_channel

    gap_fill = StackedGapFillAuthority(**authority.gap_fill.materializer_kwargs())
    assert _json_ready(gap_fill.schedule_receipt) == _json_ready(
        stacked_gap_fill_producer_schedule_receipt()
    )

    primary = StackedPrimaryQrfAuthority(
        **authority.primary_qrf.materializer_kwargs()
    )
    assert primary.target_order == PRIMARY_QRF_TARGET_ORDER
    late = StackedLateProducerAuthority(
        **authority.late_producers.materializer_kwargs()
    )
    assert late.schedule == CANONICAL_US_LATE_PRODUCER_SCHEDULE

    terminal = materialize_stacked_terminal_authority(
        **authority.terminal.materializer_kwargs()
    )
    assert _json_ready(terminal.post_puf_transfer_surface) == _json_ready(
        CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
    )
    assert _json_ready(authority.terminal.post_puf_puf_producer_surface) == (
        _json_ready(CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE)
    )
    assert _json_ready(authority.terminal.post_puf_source_producer_surface) == (
        _json_ready(CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE)
    )


def test_factory_is_pure_and_does_not_invoke_legacy_payload(
    plan: USPoolRuntimePlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import microcosm.build.spec_engine.legacy_adapter as legacy_adapter

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("physical authority attempted ambient compatibility access")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(legacy_adapter, "compile_to_legacy_payload", forbidden)
    authority = compile_us_pool_physical_authority(plan)
    assert authority.authority_sha256 == plan.authority_sha256


def test_factory_rejects_duplicate_typed_gap_direction(
    plan: USPoolRuntimePlan,
) -> None:
    candidate = replace(plan)
    runtime = plan.imputation.runtime
    gap_rows = runtime["gap_fill_plan"]
    assert isinstance(gap_rows, tuple)
    bad_runtime = _replace_map(runtime, gap_fill_plan=(gap_rows[0], gap_rows[0]))
    object.__setattr__(candidate, "imputation", replace(plan.imputation, runtime=bad_runtime))
    object.__setattr__(candidate, "_seal_sha256", _plan_sha256(candidate))

    with pytest.raises(USPoolPhysicalAuthorityError, match="duplicate ids"):
        compile_us_pool_physical_authority(candidate)


def test_factory_rejects_non_plan() -> None:
    with pytest.raises(TypeError, match="USPoolRuntimePlan"):
        compile_us_pool_physical_authority(FrozenMap())  # type: ignore[arg-type]


def test_adapter_exposes_no_program_named_accessors(
    authority: USPoolPhysicalAuthority,
) -> None:
    program_ids = {program.id for program in authority.take_up.programs}
    public_names = set(dir(authority)) | set(dir(authority.take_up))
    assert program_ids.isdisjoint(public_names)
