"""Typed physical authorities materialized from the sealed US pool plan.

This module is the bundle-mode side of the physical executor seam.  It starts
with :class:`USPoolRuntimePlan` and never imports constants-era registries,
packaged compatibility JSON, or the generated legacy payload.  The resulting
objects retain the compiler-owned ``FrozenMap`` records that define their wire
identity while adding the small typed views needed by the pool orchestrator.

The adapters intentionally do not install themselves as ambient global
configuration.  A physical kernel must accept the relevant authority as an
argument before bundle mode can dispatch it; a kernel that still selects a
private canonical registry is an integration gap, not permission to recreate
that registry here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from microcosm.build.spec_engine.canonical import canonical_json_bytes, sha256_json
from microcosm.build.spec_engine.compiler_ir import CompiledNode, SeedSiteIR
from microcosm.build.spec_engine.model import FrozenMap, FrozenValue, freeze_json
from microcosm.build.us_runtime.late_producer_dag import (
    ProducerContract,
    ProducerInput,
    ProducerInputColumn,
    ProducerOutput,
    ProducerSchedule,
    derive_producer_schedule,
)
from microcosm.build.us_runtime.pool_runtime_plan import (
    USPoolRuntimePlan,
    USPoolRuntimePlanError,
)


class USPoolPhysicalAuthorityError(ValueError):
    """The sealed pool plan cannot produce a closed physical authority."""


def _map(value: FrozenValue | object, *, location: str) -> FrozenMap:
    if not isinstance(value, FrozenMap):
        raise USPoolPhysicalAuthorityError(
            f"{location}: compiler-sealed object required"
        )
    return value


def _rows(
    value: FrozenValue | object,
    *,
    location: str,
    nonempty: bool = False,
) -> tuple[FrozenMap, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(row, FrozenMap) for row in value
    ):
        raise USPoolPhysicalAuthorityError(
            f"{location}: compiler-sealed object array required"
        )
    if nonempty and not value:
        raise USPoolPhysicalAuthorityError(f"{location}: at least one row required")
    return value


def _strings(
    value: FrozenValue | object,
    *,
    location: str,
    nonempty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise USPoolPhysicalAuthorityError(
            f"{location}: non-empty string array required"
        )
    if nonempty and not value:
        raise USPoolPhysicalAuthorityError(f"{location}: at least one value required")
    return value


def _string(value: FrozenValue | object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise USPoolPhysicalAuthorityError(f"{location}: non-empty string required")
    return value


def _integer(
    value: FrozenValue | object,
    *,
    location: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise USPoolPhysicalAuthorityError(
            f"{location}: integer >= {minimum} required"
        )
    return value


def _number(value: FrozenValue | object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise USPoolPhysicalAuthorityError(f"{location}: finite number required")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise USPoolPhysicalAuthorityError(f"{location}: finite number required")
    return result


def _optional_string(value: FrozenValue | object, *, location: str) -> str | None:
    if value is None:
        return None
    return _string(value, location=location)


def _unique(values: tuple[str, ...], *, location: str) -> None:
    duplicates = sorted(
        value for value, count in Counter(values).items() if count > 1
    )
    if duplicates:
        raise USPoolPhysicalAuthorityError(
            f"{location}: duplicate ids are forbidden: {duplicates}"
        )


def _same(left: object, right: object, *, location: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise USPoolPhysicalAuthorityError(
            f"{location}: independently compiled authorities differ"
        )


def _one[T](values: tuple[T, ...], *, location: str) -> T:
    if len(values) != 1:
        raise USPoolPhysicalAuthorityError(
            f"{location}: exactly one typed match required; matched {len(values)}"
        )
    return values[0]


@dataclass(frozen=True, slots=True)
class PhysicalGapFillAbsenceRule:
    """One exact structural-absence rule, preserving its compiler record."""

    rule_id: str
    entity: str
    column: str
    selection: str
    reason: str
    contract: FrozenMap

    def to_wire(self) -> FrozenMap:
        return self.contract


@dataclass(frozen=True, slots=True)
class PhysicalGapFillDirection:
    """One ordered cross-channel gap-fill direction."""

    name: str
    recipient_channel: str
    donor_channel: str
    target_families: FrozenMap
    recipient_absence_rules: tuple[PhysicalGapFillAbsenceRule, ...]
    transfer_execution_contract: FrozenMap
    contract: FrozenMap

    def to_wire(self) -> FrozenMap:
        return self.contract


@dataclass(frozen=True, slots=True)
class GapFillPhysicalAuthority:
    """The executable early transfer plan and its byte-stable schedule."""

    directions: tuple[PhysicalGapFillDirection, ...]
    schedule_receipt: FrozenMap

    def plan_wire(self) -> tuple[FrozenMap, ...]:
        return tuple(direction.to_wire() for direction in self.directions)

    def materializer_kwargs(self) -> dict[str, object]:
        """Return the structural inputs accepted by ``StackedGapFillAuthority``."""

        return {
            "directions": self.directions,
            "schedule_receipt": self.schedule_receipt,
        }


@dataclass(frozen=True, slots=True)
class AssemblyPhysicalAuthority:
    """Compiler-owned controls for physically assembling the stacked spine."""

    household_mass_shares: FrozenMap
    mass_anchor_channel: str

    def materializer_kwargs(self) -> dict[str, object]:
        """Return the structural inputs accepted by ``StackedAssemblyAuthority``."""

        return {
            "household_mass_shares": self.household_mass_shares,
            "mass_anchor_channel": self.mass_anchor_channel,
        }


@dataclass(frozen=True, slots=True)
class TransferModelParameters:
    """Resolved QRF parameters shared by primary and ACS transfer kernels."""

    model_seed: int
    primary_n_estimators: int
    transfer_n_estimators: int
    max_targets_per_fit: int
    transfer_execution_base: FrozenMap


@dataclass(frozen=True, slots=True)
class PrimaryQRFPhysicalAuthority:
    """Primary PUF QRF node, target order, predictors, and checkpoint ABI."""

    node: CompiledNode
    predictors: tuple[str, ...]
    person_outputs: tuple[str, ...]
    tax_unit_outputs: tuple[str, ...]
    target_order: tuple[str, ...]
    target_order_sha256: str
    checkpoint_schema_version: int
    manifest_filename: str
    contract: FrozenMap

    def materializer_kwargs(self) -> dict[str, object]:
        """Return the structural inputs accepted by ``StackedPrimaryQrfAuthority``."""

        return {
            "predictors": self.predictors,
            "person_outputs": self.person_outputs,
            "tax_unit_outputs": self.tax_unit_outputs,
            "target_order": self.target_order,
            "target_order_sha256": self.target_order_sha256,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "manifest_filename": self.manifest_filename,
        }


@dataclass(frozen=True, slots=True)
class LateTransferGroup:
    """One compiler-declared bounded transfer node in authored order."""

    name: str
    entity: str
    family: str
    targets: tuple[str, ...]
    target_families: FrozenMap
    transfer_execution_contract: FrozenMap
    node: CompiledNode
    contract: FrozenMap

    def schedule_wire(self) -> FrozenMap:
        return self.contract


@dataclass(frozen=True, slots=True)
class LateProducerPhysicalAuthority:
    """Full producer schedule and the bounded late-transfer subset."""

    schedule_receipt: FrozenMap
    producer_order: tuple[str, ...]
    waves: tuple[tuple[str, ...], ...]
    nodes: tuple[CompiledNode, ...]
    registry: Mapping[str, ProducerContract]
    schedule: ProducerSchedule
    transfer_groups: tuple[LateTransferGroup, ...]
    source_output_order: tuple[tuple[str, str], ...]
    overlap_ownership: FrozenMap
    resource_semantics: FrozenMap

    def transfer_group_wire(self) -> tuple[FrozenMap, ...]:
        return tuple(group.schedule_wire() for group in self.transfer_groups)

    def materializer_kwargs(self) -> dict[str, object]:
        """Return the structural inputs accepted by ``StackedLateProducerAuthority``."""

        return {
            "registry": self.registry,
            "schedule": self.schedule,
            "schedule_receipt": self.schedule_receipt,
            "transfer_groups": self.transfer_groups,
        }


@dataclass(frozen=True, slots=True)
class SeedPhysicalAuthority:
    """Resolved model, sampling, and attachment defaults plus ledger sites."""

    protocol_id: str
    implementation_sha256: str
    model_seed: int
    sampling_seed: int
    sampling_channels: tuple[str, ...]
    sampling_streams: FrozenMap
    model_sites: tuple[SeedSiteIR, ...]
    sampling_sites: tuple[SeedSiteIR, ...]
    clone_attachment_seed: int
    clone_attachment_stream: str


@dataclass(frozen=True, slots=True)
class TakeUpStepAuthority:
    """One typed step in one compiler-authored take-up pipeline."""

    program_id: str
    program_variable: str
    segment_ordinal: int | None
    step_ordinal: int
    kind: str
    kernel: str
    contract: FrozenMap


@dataclass(frozen=True, slots=True)
class TakeUpProgramAuthority:
    """One authored program joined injectively to its runtime engine facts."""

    id: str
    variable: str
    contract: FrozenMap
    runtime_contract: FrozenMap
    steps: tuple[TakeUpStepAuthority, ...]


@dataclass(frozen=True, slots=True)
class TakeUpPhysicalAuthority:
    """Typed programs and exact constants-compatible contract identity."""

    contract: FrozenMap
    runtime_contract: FrozenMap
    identity: FrozenMap
    programs: tuple[TakeUpProgramAuthority, ...]
    steps: tuple[TakeUpStepAuthority, ...]
    program_bindings: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True, slots=True, order=True)
class RemainingStageInputAuthority:
    """One statically declared read after the transferred checkpoint."""

    stage: str
    consumer: str
    entity: str
    variable: str
    execution_scope: str
    provision: str
    available_by: str
    fallback: str | None
    contract: FrozenMap


@dataclass(frozen=True, slots=True)
class RemainingStagePhysicalAuthority:
    """Post-transfer input registry plus physical operation schedules."""

    engine: FrozenMap
    manifest: FrozenMap
    manifest_receipt: FrozenMap
    checkpoint_identity_receipt: FrozenMap
    inputs: tuple[RemainingStageInputAuthority, ...]
    stage_order: tuple[str, ...]
    pre_clone_source_operator_order: tuple[str, ...]
    post_clone_source_operator_order: tuple[str, ...]
    derive_operator_order: tuple[str, ...]

    def require_inputs(
        self,
        *,
        stage: str,
        consumer: str,
    ) -> tuple[RemainingStageInputAuthority, ...]:
        matches = tuple(
            row
            for row in self.inputs
            if row.stage == stage and row.consumer == consumer
        )
        if not matches:
            raise USPoolPhysicalAuthorityError(
                "remaining_stage/rows: no inputs for "
                f"stage={stage!r}, consumer={consumer!r}"
            )
        return matches


@dataclass(frozen=True, slots=True)
class BatteryGateAuthority:
    """One compiler-authored terminal gate."""

    id: str
    kernel: str
    contract: FrozenMap


@dataclass(frozen=True, slots=True)
class BatteryPhysicalAuthority:
    """Executable gate contract and compiler-resolved registries."""

    contract: FrozenMap
    runtime_contract: FrozenMap
    components: FrozenMap
    gates: tuple[BatteryGateAuthority, ...]
    thresholds: FrozenMap
    declared_surface: FrozenMap
    metric_registry: tuple[FrozenMap, ...]
    joint_metric_registry: tuple[FrozenMap, ...]
    support_profile: FrozenMap


@dataclass(frozen=True, slots=True)
class TerminalPhysicalAuthority:
    """Compiler projection consumed by gap-fill and terminal stacked kernels."""

    gap_fill_directions: tuple[PhysicalGapFillDirection, ...]
    post_puf_transfer_surface: FrozenMap
    post_puf_puf_producer_surface: FrozenMap
    post_puf_source_producer_surface: FrozenMap
    declared_surface: FrozenMap
    metric_registry: tuple[FrozenMap, ...]
    joint_metric_registry: tuple[FrozenMap, ...]
    support_profile: FrozenMap
    puf_capital_gains_tail_support_contract: FrozenMap
    late_producer_schedule: FrozenMap
    compatibility_receipt: FrozenMap

    def materializer_kwargs(self) -> dict[str, object]:
        """Return exact inputs for ``materialize_stacked_terminal_authority``."""

        return {
            "gap_fill_directions": self.gap_fill_directions,
            "post_puf_transfer_surface": self.post_puf_transfer_surface,
            "post_puf_puf_producer_surface": (
                self.post_puf_puf_producer_surface
            ),
            "post_puf_source_producer_surface": (
                self.post_puf_source_producer_surface
            ),
            "declared_surface": self.declared_surface,
            "metric_registry": self.metric_registry,
            "joint_metric_registry": self.joint_metric_registry,
            "support_profile": self.support_profile,
            "puf_capital_gains_tail_support_contract": (
                self.puf_capital_gains_tail_support_contract
            ),
            "late_producer_schedule": self.late_producer_schedule,
            "compatibility_receipt": self.compatibility_receipt,
        }


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    """Physical pipeline settings that used to be module constants."""

    target_period: int
    model_seed: int
    sampling_seed: int
    household_batch_size: int
    operator_order: tuple[str, ...]
    exact_count_rule: str
    sample_fraction_default: float
    sample_rungs: tuple[FrozenMap, ...]
    sampling_channels: tuple[str, ...]
    clone_attachment_fraction_default: float
    household_mass_shares: FrozenMap
    mass_anchor_channel: str


@dataclass(frozen=True, slots=True)
class USPoolPhysicalAuthority:
    """Closed bundle-mode authority consumed by US pool physical kernels."""

    authority_sha256: str
    spec_sha256: str
    assembly: AssemblyPhysicalAuthority
    gap_fill: GapFillPhysicalAuthority
    model: TransferModelParameters
    primary_qrf: PrimaryQRFPhysicalAuthority
    late_producers: LateProducerPhysicalAuthority
    seeds: SeedPhysicalAuthority
    take_up: TakeUpPhysicalAuthority
    remaining_stage: RemainingStagePhysicalAuthority
    battery: BatteryPhysicalAuthority
    terminal: TerminalPhysicalAuthority
    simulation: SimulationSettings

    @classmethod
    def from_runtime_plan(cls, plan: USPoolRuntimePlan) -> USPoolPhysicalAuthority:
        """Materialize physical authorities from exactly one sealed plan."""

        if not isinstance(plan, USPoolRuntimePlan):
            raise TypeError(
                "USPoolPhysicalAuthority.from_runtime_plan requires "
                "USPoolRuntimePlan"
            )
        try:
            plan.__post_init__()
        except USPoolRuntimePlanError as error:
            raise USPoolPhysicalAuthorityError(
                "runtime plan seal is invalid"
            ) from error

        static = _map(
            plan.execution.checkpoint_static_components,
            location="execution/checkpoint_static_components",
        )
        pool_code = _map(
            static.get("pool_code"),
            location="execution/checkpoint_static_components/pool_code",
        )
        assembly = _compile_assembly(plan)
        gap_fill = _compile_gap_fill(plan)
        model = _compile_model(plan, static=static, pool_code=pool_code)
        primary_qrf = _compile_primary_qrf(plan, pool_code=pool_code)
        late_producers = _compile_late_producers(plan, pool_code=pool_code)
        seeds = _compile_seeds(plan, model_seed=model.model_seed)
        battery = _compile_battery(plan)
        return cls(
            authority_sha256=plan.authority_sha256,
            spec_sha256=plan.spec_sha256,
            assembly=assembly,
            gap_fill=gap_fill,
            model=model,
            primary_qrf=primary_qrf,
            late_producers=late_producers,
            seeds=seeds,
            take_up=_compile_take_up(plan, pool_code=pool_code),
            remaining_stage=_compile_remaining_stage(plan, pool_code=pool_code),
            battery=battery,
            terminal=_compile_terminal(
                plan,
                pool_code=pool_code,
                gap_fill=gap_fill,
                primary_qrf=primary_qrf,
                late_producers=late_producers,
                battery=battery,
            ),
            simulation=_compile_simulation(
                plan,
                assembly=assembly,
                static=static,
                pool_code=pool_code,
                seeds=seeds,
            ),
        )


def _compile_assembly(plan: USPoolRuntimePlan) -> AssemblyPhysicalAuthority:
    assembly = plan.assembly_sampling.assembly
    shares = _map(
        assembly.get("household_mass_shares"),
        location="spine/assembly/household_mass_shares",
    )
    if not shares:
        raise USPoolPhysicalAuthorityError(
            "spine/assembly/household_mass_shares: non-empty object required"
        )
    for channel, value in shares.entries:
        _string(channel, location="spine/assembly/household_mass_shares/channel")
        fraction = _number(
            value,
            location=f"spine/assembly/household_mass_shares/{channel}",
        )
        if not 0 <= fraction <= 1:
            raise USPoolPhysicalAuthorityError(
                "spine/assembly/household_mass_shares: shares must be in [0, 1]"
            )
    if sum(float(value) for _, value in shares.entries) != 1.0:
        raise USPoolPhysicalAuthorityError(
            "spine/assembly/household_mass_shares: shares must sum to one"
        )
    anchor = _string(
        assembly.get("mass_anchor_channel"),
        location="spine/assembly/mass_anchor_channel",
    )
    if anchor not in shares:
        raise USPoolPhysicalAuthorityError(
            "spine/assembly/mass_anchor_channel: channel has no mass share"
        )
    return AssemblyPhysicalAuthority(
        household_mass_shares=shares,
        mass_anchor_channel=anchor,
    )


def _compile_gap_fill(plan: USPoolRuntimePlan) -> GapFillPhysicalAuthority:
    runtime = plan.imputation.runtime
    plan_rows = _rows(
        runtime.get("gap_fill_plan"),
        location="imputation/runtime/gap_fill_plan",
        nonempty=True,
    )
    schedule = _map(
        runtime.get("gap_fill_producer_schedule_receipt"),
        location="imputation/runtime/gap_fill_producer_schedule_receipt",
    )
    schedule_rows = _rows(
        schedule.get("directions"),
        location="imputation/runtime/gap_fill_producer_schedule_receipt/directions",
        nonempty=True,
    )
    execution = _map(
        runtime.get("transfer_execution_contract_identities"),
        location="imputation/runtime/transfer_execution_contract_identities",
    )
    early_contracts = _rows(
        execution.get("early"),
        location="imputation/runtime/transfer_execution_contract_identities/early",
        nonempty=True,
    )

    names = tuple(
        _string(row.get("name"), location=f"gap_fill_plan/{index}/name")
        for index, row in enumerate(plan_rows)
    )
    _unique(names, location="imputation/runtime/gap_fill_plan")
    schedule_names = tuple(
        _string(
            row.get("name"),
            location=f"gap_fill_schedule/directions/{index}/name",
        )
        for index, row in enumerate(schedule_rows)
    )
    if schedule_names != names:
        raise USPoolPhysicalAuthorityError(
            "gap-fill schedule order differs from the materialized plan"
        )
    if _integer(
        schedule.get("direction_count"),
        location="gap_fill_schedule/direction_count",
        minimum=1,
    ) != len(plan_rows):
        raise USPoolPhysicalAuthorityError(
            "gap-fill schedule direction count differs from its rows"
        )

    directions: list[PhysicalGapFillDirection] = []
    for index, row in enumerate(plan_rows):
        name = names[index]
        execution_contract = _one(
            tuple(item for item in early_contracts if item.get("direction") == name),
            location=f"gap_fill_plan/{name}/transfer_execution_contract",
        )
        target_families = _map(
            row.get("target_families"),
            location=f"gap_fill_plan/{name}/target_families",
        )
        schedule_targets = _rows(
            schedule_rows[index].get("targets"),
            location=f"gap_fill_schedule/directions/{name}/targets",
            nonempty=True,
        )
        ordered_targets = tuple(
            _string(
                target.get("column"),
                location=f"gap_fill_schedule/directions/{name}/targets/column",
            )
            for target in schedule_targets
        )
        scheduled_keys = {
            (
                _string(
                    target.get("entity"),
                    location=(
                        f"gap_fill_schedule/directions/{name}/targets/entity"
                    ),
                ),
                _string(
                    target.get("family"),
                    location=(
                        f"gap_fill_schedule/directions/{name}/targets/family"
                    ),
                ),
                ordered_targets[target_index],
            )
            for target_index, target in enumerate(schedule_targets)
        }
        surface_keys = {
            (entity, family, target)
            for entity, families_value in target_families.entries
            for families in [
                _map(
                    families_value,
                    location=f"gap_fill_plan/{name}/target_families/{entity}",
                )
            ]
            for family, targets_value in families.entries
            for target in _strings(
                targets_value,
                location=(
                    f"gap_fill_plan/{name}/target_families/{entity}/{family}"
                ),
                nonempty=True,
            )
        }
        if len(scheduled_keys) != len(schedule_targets) or scheduled_keys != surface_keys:
            raise USPoolPhysicalAuthorityError(
                f"gap-fill direction {name!r} schedule target surface changed"
            )
        if _strings(
            execution_contract.get("ordered_targets"),
            location=f"gap_fill_plan/{name}/execution_contract/ordered_targets",
            nonempty=True,
        ) != ordered_targets:
            raise USPoolPhysicalAuthorityError(
                f"gap-fill direction {name!r} execution target order changed"
            )
        raw_rules = _rows(
            row.get("recipient_absence_rules"),
            location=f"gap_fill_plan/{name}/recipient_absence_rules",
        )
        rules = tuple(
            PhysicalGapFillAbsenceRule(
                rule_id=_string(
                    rule.get("rule_id"),
                    location=f"gap_fill_plan/{name}/absence_rules/{rule_index}/rule_id",
                ),
                entity=_string(
                    rule.get("entity"),
                    location=f"gap_fill_plan/{name}/absence_rules/{rule_index}/entity",
                ),
                column=_string(
                    rule.get("column"),
                    location=f"gap_fill_plan/{name}/absence_rules/{rule_index}/column",
                ),
                selection=_string(
                    rule.get("selection"),
                    location=(
                        f"gap_fill_plan/{name}/absence_rules/{rule_index}/selection"
                    ),
                ),
                reason=_string(
                    rule.get("reason"),
                    location=f"gap_fill_plan/{name}/absence_rules/{rule_index}/reason",
                ),
                contract=rule,
            )
            for rule_index, rule in enumerate(raw_rules)
        )
        rule_keys = tuple(f"{rule.entity}.{rule.column}" for rule in rules)
        _unique(rule_keys, location=f"gap_fill_plan/{name}/recipient_absence_rules")
        directions.append(
            PhysicalGapFillDirection(
                name=name,
                recipient_channel=_string(
                    row.get("recipient_channel"),
                    location=f"gap_fill_plan/{name}/recipient_channel",
                ),
                donor_channel=_string(
                    row.get("donor_channel"),
                    location=f"gap_fill_plan/{name}/donor_channel",
                ),
                target_families=target_families,
                recipient_absence_rules=rules,
                transfer_execution_contract=_map(
                    execution_contract.get("identity"),
                    location=f"gap_fill_plan/{name}/execution_contract/identity",
                ),
                contract=row,
            )
        )
    return GapFillPhysicalAuthority(tuple(directions), schedule)


def _compile_model(
    plan: USPoolRuntimePlan,
    *,
    static: FrozenMap,
    pool_code: FrozenMap,
) -> TransferModelParameters:
    models = _map(
        plan.imputation.contract.get("models"),
        location="imputation/models",
    )
    candidates = tuple(
        (model_id, _map(value, location=f"imputation/models/{model_id}"))
        for model_id, value in models.entries
        if isinstance(value, FrozenMap)
        and isinstance(value.get("kernel"), str)
        and value.get("kernel") == "kernel:regime_gated_qrf"
    )
    _model_id, model = _one(candidates, location="imputation/models/QRF kernel")
    params = _map(model.get("params"), location="imputation/models/QRF/params")
    n_estimators = _integer(
        params.get("n_estimators"),
        location="imputation/models/QRF/params/n_estimators",
        minimum=1,
    )
    family_rows = _rows(
        plan.imputation.contract.get("families"),
        location="imputation/families",
        nonempty=True,
    )
    target_limits = {
        _integer(
            row.get("max_targets_per_fit"),
            location=f"imputation/families/{index}/max_targets_per_fit",
            minimum=1,
        )
        for index, row in enumerate(family_rows)
        if row.get("max_targets_per_fit") is not None
    }
    if len(target_limits) != 1:
        raise USPoolPhysicalAuthorityError(
            "imputation/families: exactly one shared non-null "
            "max_targets_per_fit required"
        )
    target_limit = target_limits.pop()
    model_seed = _integer(
        static.get("model_seed"),
        location="checkpoint_static_components/model_seed",
    )
    primary_estimators = _integer(
        pool_code.get("primary_qrf_n_estimators"),
        location="checkpoint_static_components/pool_code/primary_qrf_n_estimators",
        minimum=1,
    )
    transfer_estimators = _integer(
        pool_code.get("acs_transfer_n_estimators"),
        location="checkpoint_static_components/pool_code/acs_transfer_n_estimators",
        minimum=1,
    )
    static_target_limit = _integer(
        pool_code.get("acs_transfer_max_targets_per_fit"),
        location=(
            "checkpoint_static_components/pool_code/"
            "acs_transfer_max_targets_per_fit"
        ),
        minimum=1,
    )
    if (
        primary_estimators != n_estimators
        or transfer_estimators != n_estimators
        or static_target_limit != target_limit
    ):
        raise USPoolPhysicalAuthorityError(
            "checkpoint model parameters differ from normalized imputation"
        )
    execution = _map(
        plan.imputation.runtime.get("transfer_execution_contract_identities"),
        location="imputation/runtime/transfer_execution_contract_identities",
    )
    return TransferModelParameters(
        model_seed=model_seed,
        primary_n_estimators=primary_estimators,
        transfer_n_estimators=transfer_estimators,
        max_targets_per_fit=static_target_limit,
        transfer_execution_base=_map(
            execution.get("base"),
            location="transfer_execution_contract_identities/base",
        ),
    )


def _compile_primary_qrf(
    plan: USPoolRuntimePlan,
    *,
    pool_code: FrozenMap,
) -> PrimaryQRFPhysicalAuthority:
    contract = _map(
        plan.imputation.runtime.get("primary_qrf"),
        location="imputation/runtime/primary_qrf",
    )
    predictors = _strings(
        contract.get("predictors"),
        location="imputation/runtime/primary_qrf/predictors",
        nonempty=True,
    )
    person_outputs = _strings(
        contract.get("person_outputs"),
        location="imputation/runtime/primary_qrf/person_outputs",
        nonempty=True,
    )
    tax_unit_outputs = _strings(
        contract.get("tax_unit_outputs"),
        location="imputation/runtime/primary_qrf/tax_unit_outputs",
        nonempty=True,
    )
    target_order = _strings(
        contract.get("target_order"),
        location="imputation/runtime/primary_qrf/target_order",
        nonempty=True,
    )
    if target_order != person_outputs + tax_unit_outputs:
        raise USPoolPhysicalAuthorityError(
            "primary QRF target order differs from person plus tax-unit outputs"
        )
    _unique(target_order, location="imputation/runtime/primary_qrf/target_order")
    expected_outputs = tuple(("person", name) for name in person_outputs) + tuple(
        ("tax_unit", name) for name in tax_unit_outputs
    )
    authored_nodes = _rows(
        plan.imputation.producer_graph.get("nodes"),
        location="imputation/producer_graph/nodes",
        nonempty=True,
    )
    primary_rows = tuple(
        row for row in authored_nodes if row.get("kind") == "primary_puf"
    )
    primary_row = _one(primary_rows, location="producer graph/primary QRF kind")
    node = plan.imputation.require_node(
        _string(primary_row.get("id"), location="producer graph/primary QRF id")
    )
    node_source_path = f"/imputation/producer_graph/nodes/{node.id}"
    node_source_param = _one(
        tuple(param for param in node.resolved_params if param.path == node_source_path),
        location=f"compiled node {node.id!r} authored source",
    )
    node_source = _map(
        node_source_param.value,
        location=f"compiled node {node.id!r} authored source",
    )
    if _string(
        node_source.get("id"),
        location=f"compiled node {node.id!r} resolved source id",
    ) != node.id:
        raise USPoolPhysicalAuthorityError(
            f"compiled node {node.id!r} resolved source id changed"
        )
    virtual_resources = _rows(
        node_source.get("virtual_resources"),
        location=f"compiled node {node.id!r}/virtual_resources",
        nonempty=True,
    )
    checkpoint_bindings: list[FrozenMap] = []
    for resource_index, resource in enumerate(virtual_resources):
        binding = _map(
            resource.get("binding"),
            location=(
                f"compiled node {node.id!r}/virtual_resources/"
                f"{resource_index}/binding"
            ),
        )
        if binding.get("resource_kind") != "primary_qrf_checkpoint":
            continue
        if resource.get("kind") != "target_bank":
            raise USPoolPhysicalAuthorityError(
                "primary QRF checkpoint binding is not a target-bank resource"
            )
        checkpoint_bindings.append(binding)
    checkpoint_binding = _one(
        tuple(checkpoint_bindings),
        location=f"compiled node {node.id!r} primary QRF checkpoint binding",
    )
    manifest_filename = _string(
        checkpoint_binding.get("manifest_filename"),
        location=f"compiled node {node.id!r} checkpoint manifest filename",
    )
    node_outputs = {
        (row.get("entity"), row.get("column")) for row in node.outputs
    }
    missing_outputs = tuple(
        item for item in expected_outputs if item not in node_outputs
    )
    if missing_outputs:
        raise USPoolPhysicalAuthorityError(
            "primary QRF target surface is absent from its compiled node: "
            f"{missing_outputs}"
        )
    if tuple(site.id for site in node.seed_sites) != tuple(
        dict.fromkeys(site.id for site in node.seed_sites)
    ):
        raise USPoolPhysicalAuthorityError("primary QRF node repeats seed sites")
    checkpoint_version = _integer(
        contract.get("checkpoint_schema_version"),
        location="imputation/runtime/primary_qrf/checkpoint_schema_version",
        minimum=1,
    )
    _integer(
        checkpoint_binding.get("schema_version"),
        location=f"compiled node {node.id!r} resource schema version",
        minimum=1,
    )
    if _strings(
        pool_code.get("primary_qrf_target_order"),
        location="checkpoint_static_components/pool_code/primary_qrf_target_order",
        nonempty=True,
    ) != target_order:
        raise USPoolPhysicalAuthorityError(
            "checkpoint primary QRF target order differs from imputation authority"
        )
    if _integer(
        pool_code.get("primary_qrf_checkpoint_schema_version"),
        location=(
            "checkpoint_static_components/pool_code/"
            "primary_qrf_checkpoint_schema_version"
        ),
        minimum=1,
    ) != checkpoint_version:
        raise USPoolPhysicalAuthorityError(
            "checkpoint primary QRF schema differs from imputation authority"
        )
    digest = _string(
        contract.get("target_order_sha256"),
        location="imputation/runtime/primary_qrf/target_order_sha256",
    )
    if sha256_json(target_order) != digest:
        raise USPoolPhysicalAuthorityError(
            "primary QRF target-order digest does not match its order"
        )
    return PrimaryQRFPhysicalAuthority(
        node=node,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
        target_order=target_order,
        target_order_sha256=digest,
        checkpoint_schema_version=checkpoint_version,
        manifest_filename=manifest_filename,
        contract=contract,
    )


def _producer_input_column(
    row: FrozenMap,
    *,
    location: str,
) -> ProducerInputColumn:
    return ProducerInputColumn(
        entity=_string(row.get("entity"), location=f"{location}/entity"),
        column=_string(row.get("column"), location=f"{location}/column"),
        value_kind=_string(
            row.get("value_kind"), location=f"{location}/value_kind"
        ),
    )


def _producer_input(row: FrozenMap, *, location: str) -> ProducerInput:
    raw_alternatives = row.get("alternatives")
    if not isinstance(raw_alternatives, tuple) or not raw_alternatives:
        raise USPoolPhysicalAuthorityError(
            f"{location}/alternatives: non-empty matrix required"
        )
    alternatives: list[tuple[ProducerInputColumn, ...]] = []
    for alternative_index, raw_alternative in enumerate(raw_alternatives):
        if not isinstance(raw_alternative, tuple) or not raw_alternative:
            raise USPoolPhysicalAuthorityError(
                f"{location}/alternatives/{alternative_index}: "
                "non-empty object array required"
            )
        alternatives.append(
            tuple(
                _producer_input_column(
                    _map(
                        item,
                        location=(
                            f"{location}/alternatives/{alternative_index}/"
                            f"{item_index}"
                        ),
                    ),
                    location=(
                        f"{location}/alternatives/{alternative_index}/{item_index}"
                    ),
                )
                for item_index, item in enumerate(raw_alternative)
            )
        )
    return ProducerInput(
        entity=_string(row.get("entity"), location=f"{location}/entity"),
        column=_string(row.get("column"), location=f"{location}/column"),
        required_scope=_string(
            row.get("required_scope"), location=f"{location}/required_scope"
        ),
        producing_stage=_string(
            row.get("producing_stage"), location=f"{location}/producing_stage"
        ),
        tolerated_absence_receipts=_strings(
            row.get("tolerated_absence_receipts"),
            location=f"{location}/tolerated_absence_receipts",
        ),
        alternatives=tuple(alternatives),
    )


def _producer_output(row: FrozenMap, *, location: str) -> ProducerOutput:
    return ProducerOutput(
        entity=_string(row.get("entity"), location=f"{location}/entity"),
        column=_string(row.get("column"), location=f"{location}/column"),
        coverage_scope=_string(
            row.get("coverage_scope"), location=f"{location}/coverage_scope"
        ),
    )


def _materialize_producer_registry(
    plan: USPoolRuntimePlan,
) -> Mapping[str, ProducerContract]:
    authored_rows = _rows(
        plan.imputation.producer_graph.get("nodes"),
        location="imputation/producer_graph/nodes",
        nonempty=True,
    )
    authored_by_id: dict[str, FrozenMap] = {}
    for index, row in enumerate(authored_rows):
        node_id = _string(
            row.get("id"), location=f"imputation/producer_graph/nodes/{index}/id"
        )
        if node_id in authored_by_id:
            raise USPoolPhysicalAuthorityError(
                f"imputation/producer_graph/nodes: duplicate id {node_id!r}"
            )
        authored_by_id[node_id] = row
    if set(authored_by_id) != {node.id for node in plan.imputation.nodes}:
        raise USPoolPhysicalAuthorityError(
            "authored and compiled producer node ids differ"
        )

    registry: dict[str, ProducerContract] = {}
    for node in plan.imputation.nodes:
        authored = authored_by_id[node.id]
        registry[node.id] = ProducerContract(
            name=node.id,
            kind=_string(
                authored.get("kind"),
                location=f"imputation/producer_graph/nodes/{node.id}/kind",
            ),
            inputs=tuple(
                _producer_input(
                    row,
                    location=f"compiled/nodes/{node.id}/inputs/{index}",
                )
                for index, row in enumerate(node.inputs)
            ),
            outputs=tuple(
                _producer_output(
                    row,
                    location=f"compiled/nodes/{node.id}/outputs/{index}",
                )
                for index, row in enumerate(node.outputs)
            ),
        )
    return MappingProxyType(registry)


def _compile_late_producers(
    plan: USPoolRuntimePlan,
    *,
    pool_code: FrozenMap,
) -> LateProducerPhysicalAuthority:
    runtime = plan.imputation.runtime
    schedule = _map(
        runtime.get("late_producer_schedule_receipt"),
        location="imputation/runtime/late_producer_schedule_receipt",
    )
    _same(
        pool_code.get("late_producer_schedule"),
        schedule,
        location="late producer checkpoint schedule",
    )
    order = _strings(
        schedule.get("order"),
        location="late_producer_schedule/order",
        nonempty=True,
    )
    _unique(order, location="late_producer_schedule/order")
    nodes = tuple(plan.imputation.require_node(node_id) for node_id in order)
    if nodes != plan.imputation.nodes:
        raise USPoolPhysicalAuthorityError(
            "late producer schedule order differs from compiled-node order"
        )
    raw_waves = schedule.get("waves")
    if not isinstance(raw_waves, tuple) or not all(
        isinstance(wave, tuple) for wave in raw_waves
    ):
        raise USPoolPhysicalAuthorityError(
            "late_producer_schedule/waves: string-array matrix required"
        )
    waves = tuple(
        _strings(
            wave,
            location=f"late_producer_schedule/waves/{index}",
            nonempty=True,
        )
        for index, wave in enumerate(raw_waves)
    )
    if tuple(node for wave in waves for node in wave) != order:
        raise USPoolPhysicalAuthorityError(
            "late producer waves do not exactly partition total order"
        )
    registry = _materialize_producer_registry(plan)
    producer_schedule = derive_producer_schedule(
        registry,
        external_stages=_strings(
            schedule.get("external_stages"),
            location="late_producer_schedule/external_stages",
        ),
    )
    if producer_schedule.order != order or producer_schedule.waves != waves:
        raise USPoolPhysicalAuthorityError(
            "materialized producer registry derives a different schedule"
        )
    raw_edges = schedule.get("edges")
    if not isinstance(raw_edges, tuple) or not all(
        isinstance(edge, tuple) for edge in raw_edges
    ):
        raise USPoolPhysicalAuthorityError(
            "late_producer_schedule/edges: pair array required"
        )
    edges = tuple(
        _strings(
            edge,
            location=f"late_producer_schedule/edges/{index}",
            nonempty=True,
        )
        for index, edge in enumerate(raw_edges)
    )
    if any(len(edge) != 2 for edge in edges):
        raise USPoolPhysicalAuthorityError(
            "late_producer_schedule/edges: two-item pairs required"
        )
    if producer_schedule.edges != edges:
        raise USPoolPhysicalAuthorityError(
            "materialized producer registry derives different edges"
        )
    if producer_schedule.sha256 != _string(
        schedule.get("schedule_sha256"),
        location="late_producer_schedule/schedule_sha256",
    ):
        raise USPoolPhysicalAuthorityError(
            "materialized producer schedule digest differs from compiler receipt"
        )

    execution = _map(
        runtime.get("transfer_execution_contract_identities"),
        location="imputation/runtime/transfer_execution_contract_identities",
    )
    late_execution = _rows(
        execution.get("late"),
        location="transfer_execution_contract_identities/late",
        nonempty=True,
    )
    group_rows = _rows(
        schedule.get("transfer_groups"),
        location="late_producer_schedule/transfer_groups",
        nonempty=True,
    )
    group_names = tuple(
        _string(row.get("name"), location=f"late transfer group/{index}/name")
        for index, row in enumerate(group_rows)
    )
    _unique(group_names, location="late_producer_schedule/transfer_groups")
    groups: list[LateTransferGroup] = []
    for index, row in enumerate(group_rows):
        name = group_names[index]
        entity = _string(row.get("entity"), location=f"late group/{name}/entity")
        family = _string(row.get("family"), location=f"late group/{name}/family")
        targets = _strings(
            row.get("targets"),
            location=f"late group/{name}/targets",
            nonempty=True,
        )
        _unique(targets, location=f"late group/{name}/targets")
        node = plan.imputation.require_node(name)
        node_outputs = tuple(
            (output.get("entity"), output.get("column"))
            for output in node.outputs
            if output.get("temporary") is False
            and output.get("validation_only") is False
        )
        expected_node_outputs = tuple((entity, target) for target in targets)
        if (
            len(node_outputs) != len(set(node_outputs))
            or set(node_outputs) != set(expected_node_outputs)
        ):
            raise USPoolPhysicalAuthorityError(
                f"late transfer group {name!r} differs from compiled node outputs"
            )
        execution_contract = _one(
            tuple(item for item in late_execution if item.get("producer") == name),
            location=f"late transfer group/{name}/execution contract",
        )
        if _strings(
            execution_contract.get("ordered_targets"),
            location=f"late transfer group/{name}/execution contract targets",
            nonempty=True,
        ) != targets:
            raise USPoolPhysicalAuthorityError(
                f"late transfer group {name!r} execution target order changed"
            )
        target_families_value = freeze_json(
            {entity: {family: list(targets)}}
        )
        assert isinstance(target_families_value, FrozenMap)
        groups.append(
            LateTransferGroup(
                name=name,
                entity=entity,
                family=family,
                targets=targets,
                target_families=target_families_value,
                transfer_execution_contract=_map(
                    execution_contract.get("identity"),
                    location=f"late transfer group/{name}/execution identity",
                ),
                node=node,
                contract=row,
            )
        )
    if _integer(
        schedule.get("transfer_group_count"),
        location="late_producer_schedule/transfer_group_count",
        minimum=1,
    ) != len(groups):
        raise USPoolPhysicalAuthorityError(
            "late producer transfer-group count differs from its rows"
        )
    if _integer(
        schedule.get("transfer_target_count"),
        location="late_producer_schedule/transfer_target_count",
        minimum=1,
    ) != sum(len(group.targets) for group in groups):
        raise USPoolPhysicalAuthorityError(
            "late producer target count differs from its groups"
        )
    if _integer(
        schedule.get("producer_count"),
        location="late_producer_schedule/producer_count",
        minimum=1,
    ) != len(nodes):
        raise USPoolPhysicalAuthorityError(
            "late producer count differs from compiled nodes"
        )
    authored_rows = _rows(
        plan.imputation.producer_graph.get("nodes"),
        location="imputation/producer_graph/nodes",
        nonempty=True,
    )
    source_output_order = tuple(
        (
            _string(
                output.get("entity"),
                location=f"producer_graph/nodes/{node_index}/outputs/entity",
            ),
            _string(
                output.get("column"),
                location=f"producer_graph/nodes/{node_index}/outputs/column",
            ),
        )
        for node_index, authored in enumerate(authored_rows)
        if authored.get("kind") == "post_clone_source"
        for output in _rows(
            authored.get("outputs"),
            location=f"producer_graph/nodes/{node_index}/outputs",
            nonempty=True,
        )
        if not str(output.get("column", "")).startswith("@")
    )
    if len(source_output_order) != len(set(source_output_order)):
        raise USPoolPhysicalAuthorityError(
            "post-clone source producers repeat physical outputs"
        )
    return LateProducerPhysicalAuthority(
        schedule_receipt=schedule,
        producer_order=order,
        waves=waves,
        nodes=nodes,
        registry=registry,
        schedule=producer_schedule,
        transfer_groups=tuple(groups),
        source_output_order=source_output_order,
        overlap_ownership=_map(
            runtime.get("overlap_ownership"),
            location="imputation/runtime/overlap_ownership",
        ),
        resource_semantics=_map(
            runtime.get("late_producer_resource_semantics"),
            location="imputation/runtime/late_producer_resource_semantics",
        ),
    )


def _seed_default(
    sites: tuple[SeedSiteIR, ...],
    *,
    value_source: str,
) -> tuple[int, tuple[SeedSiteIR, ...]]:
    matches = tuple(
        site for site in sites if site.contract.get("value_source") == value_source
    )
    if not matches:
        raise USPoolPhysicalAuthorityError(
            f"seed stream map has no {value_source!r} sites"
        )
    defaults = {
        _integer(
            site.contract.get("default"),
            location=f"seed_stream_map/sites/{site.id}/contract/default",
        )
        for site in matches
    }
    if len(defaults) != 1:
        raise USPoolPhysicalAuthorityError(
            f"seed stream map {value_source!r} sites do not share one default"
        )
    return defaults.pop(), matches


def _compile_seeds(
    plan: USPoolRuntimePlan,
    *,
    model_seed: int,
) -> SeedPhysicalAuthority:
    stream_map = plan.seed_stream_map
    model_default, model_sites = _seed_default(
        stream_map.sites,
        value_source="run_request.build_model_seed",
    )
    sampling_default, sampling_sites = _seed_default(
        stream_map.sites,
        value_source="run_request.sample_seed",
    )
    if model_default != model_seed:
        raise USPoolPhysicalAuthorityError(
            "checkpoint model seed differs from the seed stream map"
        )
    sampling = plan.assembly_sampling.runtime_sampling
    channels = _strings(
        sampling.get("channels"),
        location="sampling/channels",
        nonempty=True,
    )
    _unique(channels, location="sampling/channels")
    seed = _map(sampling.get("seed"), location="sampling/seed")
    if _integer(seed.get("default"), location="sampling/seed/default") != (
        sampling_default
    ):
        raise USPoolPhysicalAuthorityError(
            "sampling default differs from the seed stream map"
        )
    streams = _map(seed.get("streams"), location="sampling/seed/streams")
    if tuple(streams) != tuple(sorted(channels)):
        raise USPoolPhysicalAuthorityError(
            "sampling stream channels differ from the sampling inventory"
        )
    sampling_stream_ids = tuple(
        _string(streams.get(channel), location=f"sampling/seed/streams/{channel}")
        for channel in channels
    )
    site_stream_ids = tuple(f"stream:{site.stream}" for site in sampling_sites)
    if set(sampling_stream_ids) != set(site_stream_ids):
        raise USPoolPhysicalAuthorityError(
            "sampling stream bindings differ from the seed ledger"
        )

    attachment_rows: list[tuple[int, str]] = []
    for role_index, role in enumerate(plan.support_spine.roles):
        attachment_value = role.get("attachment")
        if attachment_value is None:
            continue
        attachment = _map(
            attachment_value,
            location=f"spine/support_roles/{role_index}/attachment",
        )
        attachment_seed = _map(
            attachment.get("seed"),
            location=f"spine/support_roles/{role_index}/attachment/seed",
        )
        attachment_rows.append(
            (
                _integer(
                    attachment_seed.get("default"),
                    location=(
                        f"spine/support_roles/{role_index}/attachment/seed/default"
                    ),
                ),
                _string(
                    attachment_seed.get("stream"),
                    location=(
                        f"spine/support_roles/{role_index}/attachment/seed/stream"
                    ),
                ),
            )
        )
    attachment_default, attachment_stream = _one(
        tuple(attachment_rows),
        location="spine/support_roles attachment seed",
    )
    attachment_matches = tuple(
        site
        for site in stream_map.sites
        if f"stream:{site.stream}" == attachment_stream
    )
    attachment_site = _one(
        attachment_matches,
        location="seed stream map attachment stream",
    )
    if attachment_site.contract.get("default") != attachment_default:
        raise USPoolPhysicalAuthorityError(
            "attachment default differs from the seed stream map"
        )
    return SeedPhysicalAuthority(
        protocol_id=stream_map.protocol_id,
        implementation_sha256=stream_map.implementation_sha256,
        model_seed=model_default,
        sampling_seed=sampling_default,
        sampling_channels=channels,
        sampling_streams=streams,
        model_sites=model_sites,
        sampling_sites=sampling_sites,
        clone_attachment_seed=attachment_default,
        clone_attachment_stream=attachment_stream,
    )


def _program_steps(
    *,
    program_id: str,
    variable: str,
    row: FrozenMap,
) -> tuple[TakeUpStepAuthority, ...]:
    top_pipeline = row.get("pipeline")
    segments_value = row.get("segments")
    if top_pipeline is not None and segments_value is not None:
        raise USPoolPhysicalAuthorityError(
            f"take_up/programs/{program_id}: pipeline and segments are exclusive"
        )
    pipelines: list[tuple[int | None, tuple[FrozenMap, ...]]] = []
    if top_pipeline is not None:
        pipelines.append(
            (
                None,
                _rows(
                    top_pipeline,
                    location=f"take_up/programs/{program_id}/pipeline",
                    nonempty=True,
                ),
            )
        )
    elif segments_value is not None:
        segments = _rows(
            segments_value,
            location=f"take_up/programs/{program_id}/segments",
            nonempty=True,
        )
        for segment_index, segment in enumerate(segments):
            pipelines.append(
                (
                    segment_index,
                    _rows(
                        segment.get("pipeline"),
                        location=(
                            f"take_up/programs/{program_id}/segments/"
                            f"{segment_index}/pipeline"
                        ),
                        nonempty=True,
                    ),
                )
            )
    else:
        raise USPoolPhysicalAuthorityError(
            f"take_up/programs/{program_id}: pipeline or segments required"
        )

    result: list[TakeUpStepAuthority] = []
    for segment_index, steps in pipelines:
        for step_index, step in enumerate(steps):
            result.append(
                TakeUpStepAuthority(
                    program_id=program_id,
                    program_variable=variable,
                    segment_ordinal=segment_index,
                    step_ordinal=step_index,
                    kind=_string(
                        step.get("kind"),
                        location=(
                            f"take_up/programs/{program_id}/steps/{step_index}/kind"
                        ),
                    ),
                    kernel=_string(
                        step.get("kernel"),
                        location=(
                            f"take_up/programs/{program_id}/steps/{step_index}/kernel"
                        ),
                    ),
                    contract=step,
                )
            )
    return tuple(result)


def _compile_take_up(
    plan: USPoolRuntimePlan,
    *,
    pool_code: FrozenMap,
) -> TakeUpPhysicalAuthority:
    contract = plan.take_up.contract
    runtime = plan.take_up.runtime
    identity = _map(
        pool_code.get("take_up_contract"),
        location="checkpoint_static_components/pool_code/take_up_contract",
    )
    authored_rows = _rows(
        contract.get("programs"),
        location="take_up/contract/programs",
        nonempty=True,
    )
    runtime_rows = _rows(
        runtime.get("programs"),
        location="take_up/runtime/programs",
        nonempty=True,
    )
    identity_rows = _rows(
        identity.get("programs"),
        location="take_up/identity/programs",
        nonempty=True,
    )
    _same(identity_rows, runtime_rows, location="take-up identity programs")
    if _string(identity.get("resource_sha256"), location="take_up/resource_sha256") != (
        sha256_json(runtime)
    ):
        raise USPoolPhysicalAuthorityError(
            "take-up identity resource digest differs from its runtime contract"
        )
    authored_ids = tuple(
        _string(row.get("id"), location=f"take_up/programs/{index}/id")
        for index, row in enumerate(authored_rows)
    )
    authored_variables = tuple(
        _string(
            row.get("variable"), location=f"take_up/programs/{index}/variable"
        )
        for index, row in enumerate(authored_rows)
    )
    runtime_variables = tuple(
        _string(
            row.get("variable"),
            location=f"take_up/runtime/programs/{index}/variable",
        )
        for index, row in enumerate(runtime_rows)
    )
    _unique(authored_ids, location="take_up/programs ids")
    _unique(authored_variables, location="take_up/programs variables")
    _unique(runtime_variables, location="take_up/runtime/program variables")
    if set(authored_variables) != set(runtime_variables):
        raise USPoolPhysicalAuthorityError(
            "authored and runtime take-up program variables differ"
        )

    programs: list[TakeUpProgramAuthority] = []
    all_steps: list[TakeUpStepAuthority] = []
    bindings: list[tuple[str, str, str]] = []
    for index, row in enumerate(authored_rows):
        program_id = authored_ids[index]
        variable = authored_variables[index]
        runtime_row = _one(
            tuple(item for item in runtime_rows if item.get("variable") == variable),
            location=f"take_up/programs/{program_id}/runtime contract",
        )
        steps = _program_steps(program_id=program_id, variable=variable, row=row)
        all_steps.extend(steps)
        bindings.append(
            (
                variable,
                _string(
                    runtime_row.get("entity"),
                    location=f"take_up/runtime/programs/{variable}/entity",
                ),
                _string(
                    runtime_row.get("populace_treatment"),
                    location=(
                        f"take_up/runtime/programs/{variable}/populace_treatment"
                    ),
                ),
            )
        )
        programs.append(
            TakeUpProgramAuthority(
                id=program_id,
                variable=variable,
                contract=row,
                runtime_contract=runtime_row,
                steps=steps,
            )
        )
    return TakeUpPhysicalAuthority(
        contract=contract,
        runtime_contract=runtime,
        identity=identity,
        programs=tuple(programs),
        steps=tuple(all_steps),
        program_bindings=tuple(bindings),
    )


def _compile_remaining_stage(
    plan: USPoolRuntimePlan,
    *,
    pool_code: FrozenMap,
) -> RemainingStagePhysicalAuthority:
    rows: list[RemainingStageInputAuthority] = []
    for index, row in enumerate(plan.remaining_stage.rows):
        rows.append(
            RemainingStageInputAuthority(
                stage=_string(
                    row.get("stage"), location=f"remaining_stage/rows/{index}/stage"
                ),
                consumer=_string(
                    row.get("consumer"),
                    location=f"remaining_stage/rows/{index}/consumer",
                ),
                entity=_string(
                    row.get("entity"),
                    location=f"remaining_stage/rows/{index}/entity",
                ),
                variable=_string(
                    row.get("variable"),
                    location=f"remaining_stage/rows/{index}/variable",
                ),
                execution_scope=_string(
                    row.get("execution_scope"),
                    location=f"remaining_stage/rows/{index}/execution_scope",
                ),
                provision=_string(
                    row.get("provision"),
                    location=f"remaining_stage/rows/{index}/provision",
                ),
                available_by=_string(
                    row.get("available_by"),
                    location=f"remaining_stage/rows/{index}/available_by",
                ),
                fallback=_optional_string(
                    row.get("fallback"),
                    location=f"remaining_stage/rows/{index}/fallback",
                ),
                contract=row,
            )
        )
    typed_rows = tuple(rows)
    row_keys = tuple(
        (
            row.stage,
            row.consumer,
            row.entity,
            row.variable,
            row.execution_scope,
        )
        for row in typed_rows
    )
    if len(row_keys) != len(set(row_keys)):
        raise USPoolPhysicalAuthorityError(
            "remaining_stage/rows: duplicate typed input declarations"
        )
    receipt = plan.remaining_stage.receipt
    if _integer(
        receipt.get("entry_count"),
        location="remaining_stage/receipt/entry_count",
        minimum=1,
    ) != len(typed_rows):
        raise USPoolPhysicalAuthorityError(
            "remaining-stage entry count differs from its rows"
        )
    stage_counts = _map(
        receipt.get("stage_counts"),
        location="remaining_stage/receipt/stage_counts",
    )
    consumer_counts = _map(
        receipt.get("consumer_counts"),
        location="remaining_stage/receipt/consumer_counts",
    )
    observed_stage_counts = Counter(row.stage for row in typed_rows)
    observed_consumer_counts = Counter(row.consumer for row in typed_rows)
    if dict(stage_counts.entries) != dict(observed_stage_counts):
        raise USPoolPhysicalAuthorityError(
            "remaining-stage stage counts differ from its rows"
        )
    if dict(consumer_counts.entries) != dict(observed_consumer_counts):
        raise USPoolPhysicalAuthorityError(
            "remaining-stage consumer counts differ from its rows"
        )
    stage_order = tuple(dict.fromkeys(row.stage for row in typed_rows))
    checkpoint_receipt = _map(
        pool_code.get("remaining_stage_input_manifest"),
        location=(
            "checkpoint_static_components/pool_code/"
            "remaining_stage_input_manifest"
        ),
    )
    if checkpoint_receipt.get("manifest_sha256") != receipt.get("manifest_sha256"):
        raise USPoolPhysicalAuthorityError(
            "remaining-stage checkpoint receipt differs from its manifest digest"
        )
    return RemainingStagePhysicalAuthority(
        engine=plan.remaining_stage.engine,
        manifest=plan.remaining_stage.manifest,
        manifest_receipt=receipt,
        checkpoint_identity_receipt=checkpoint_receipt,
        inputs=typed_rows,
        stage_order=stage_order,
        pre_clone_source_operator_order=_strings(
            pool_code.get("pre_clone_source_operator_order"),
            location="pool_code/pre_clone_source_operator_order",
            nonempty=True,
        ),
        post_clone_source_operator_order=_strings(
            pool_code.get("post_clone_source_operator_order"),
            location="pool_code/post_clone_source_operator_order",
            nonempty=True,
        ),
        derive_operator_order=_strings(
            pool_code.get("derive_operator_order"),
            location="pool_code/derive_operator_order",
            nonempty=True,
        ),
    )


def _metric_registry_map(
    rows: tuple[FrozenMap, ...],
    *,
    location: str,
) -> dict[tuple[str, str, str, int], str]:
    result: dict[tuple[str, str, str, int], str] = {}
    for index, row in enumerate(rows):
        key = (
            _string(row.get("entity"), location=f"{location}/{index}/entity"),
            _string(row.get("family"), location=f"{location}/{index}/family"),
            _string(row.get("column"), location=f"{location}/{index}/column"),
            _integer(
                row.get("clone_index"),
                location=f"{location}/{index}/clone_index",
            ),
        )
        metric = _string(
            row.get("metric"),
            location=f"{location}/{index}/metric",
        )
        if key in result:
            relation = "duplicate" if result[key] == metric else "conflicting"
            raise USPoolPhysicalAuthorityError(
                f"{location}: {relation} registry key {key!r}"
            )
        result[key] = metric
    return result


def _joint_metric_registry_map(
    rows: tuple[FrozenMap, ...],
    *,
    location: str,
) -> dict[tuple[str, str, tuple[str, ...], int], str]:
    result: dict[tuple[str, str, tuple[str, ...], int], str] = {}
    for index, row in enumerate(rows):
        columns = _strings(
            row.get("columns"),
            location=f"{location}/{index}/columns",
            nonempty=True,
        )
        _unique(columns, location=f"{location}/{index}/columns")
        key = (
            _string(row.get("entity"), location=f"{location}/{index}/entity"),
            _string(row.get("family"), location=f"{location}/{index}/family"),
            columns,
            _integer(
                row.get("clone_index"),
                location=f"{location}/{index}/clone_index",
            ),
        )
        metric = _string(
            row.get("metric"),
            location=f"{location}/{index}/metric",
        )
        if key in result:
            relation = "duplicate" if result[key] == metric else "conflicting"
            raise USPoolPhysicalAuthorityError(
                f"{location}: {relation} registry key {key!r}"
            )
        result[key] = metric
    return result


def _metric_registry_payload(rows: tuple[FrozenMap, ...]) -> list[dict[str, object]]:
    registry = _metric_registry_map(rows, location="battery/metric_registry")
    return [
        {
            "entity": entity,
            "family": family,
            "column": column,
            "clone_index": clone_index,
            "metric": registry[(entity, family, column, clone_index)],
        }
        for entity, family, column, clone_index in sorted(registry)
    ]


def _joint_metric_registry_payload(
    rows: tuple[FrozenMap, ...],
) -> list[dict[str, object]]:
    registry = _joint_metric_registry_map(
        rows,
        location="battery/joint_metric_registry",
    )
    return [
        {
            "entity": entity,
            "family": family,
            "columns": list(columns),
            "clone_index": clone_index,
            "metric": registry[(entity, family, columns, clone_index)],
        }
        for entity, family, columns, clone_index in sorted(registry)
    ]


def _compile_battery(plan: USPoolRuntimePlan) -> BatteryPhysicalAuthority:
    runtime = plan.battery.runtime
    raw_gates = _rows(
        runtime.get("gates"),
        location="battery/runtime/gates",
        nonempty=True,
    )
    gates = tuple(
        BatteryGateAuthority(
            id=_string(row.get("id"), location=f"battery/gates/{index}/id"),
            kernel=_string(
                row.get("kernel"), location=f"battery/gates/{index}/kernel"
            ),
            contract=row,
        )
        for index, row in enumerate(raw_gates)
    )
    _unique(tuple(gate.id for gate in gates), location="battery/gates ids")
    _unique(tuple(gate.kernel for gate in gates), location="battery/gates kernels")
    operation_ids = {operation.id for operation in plan.execution.operations}
    missing_operations = sorted(
        gate.kernel.removeprefix("kernel:")
        for gate in gates
        if gate.kernel.removeprefix("kernel:") not in operation_ids
    )
    if missing_operations:
        raise USPoolPhysicalAuthorityError(
            "battery gate kernels are absent from the physical plan: "
            f"{missing_operations}"
        )
    declared_surface = _map(
        runtime.get("declared_surface"),
        location="battery/runtime/declared_surface",
    )
    metric_registry = _rows(
        runtime.get("metric_registry"),
        location="battery/runtime/metric_registry",
        nonempty=True,
    )
    joint_registry = _rows(
        runtime.get("joint_metric_registry"),
        location="battery/runtime/joint_metric_registry",
        nonempty=True,
    )
    support_profile = _map(
        runtime.get("support_profile"),
        location="battery/runtime/support_profile",
    )
    _same(
        plan.battery.components.get("declared_surface"),
        declared_surface,
        location="battery components/declared_surface",
    )
    component_metric_registry = _rows(
        plan.battery.components.get("metric_registry"),
        location="battery/components/metric_registry",
        nonempty=True,
    )
    if _metric_registry_map(
        component_metric_registry,
        location="battery/components/metric_registry",
    ) != _metric_registry_map(
        metric_registry,
        location="battery/runtime/metric_registry",
    ):
        raise USPoolPhysicalAuthorityError(
            "battery component and runtime metric registries differ"
        )
    component_joint_registry = _rows(
        plan.battery.components.get("joint_metric_registry"),
        location="battery/components/joint_metric_registry",
        nonempty=True,
    )
    if _joint_metric_registry_map(
        component_joint_registry,
        location="battery/components/joint_metric_registry",
    ) != _joint_metric_registry_map(
        joint_registry,
        location="battery/runtime/joint_metric_registry",
    ):
        raise USPoolPhysicalAuthorityError(
            "battery component and runtime joint-metric registries differ"
        )
    component_support = _map(
        plan.battery.components.get("support_profile"),
        location="battery/components/support_profile",
    )
    for key in ("profile_id", "version", "min_effective_support"):
        if component_support.get(key) != support_profile.get(key):
            raise USPoolPhysicalAuthorityError(
                f"battery support profile component differs at {key!r}"
            )
    completeness = _map(
        runtime.get("completeness"),
        location="battery/runtime/completeness",
    )
    if _integer(
        completeness.get("targets"),
        location="battery/runtime/completeness/targets",
        minimum=1,
    ) != len(metric_registry):
        raise USPoolPhysicalAuthorityError(
            "battery completeness count differs from metric registry"
        )
    return BatteryPhysicalAuthority(
        contract=plan.battery.contract,
        runtime_contract=runtime,
        components=plan.battery.components,
        gates=gates,
        thresholds=_map(
            runtime.get("thresholds"),
            location="battery/runtime/thresholds",
        ),
        declared_surface=declared_surface,
        metric_registry=metric_registry,
        joint_metric_registry=joint_registry,
        support_profile=support_profile,
    )


def _transfer_family_base(family: str, *, location: str) -> tuple[str, int | None]:
    stem, separator, suffix = family.rpartition("__batch_")
    if not separator or not suffix.isdigit():
        return family, None
    batch = int(suffix)
    if not stem or batch < 1:
        raise USPoolPhysicalAuthorityError(
            f"{location}: malformed bounded-family suffix"
        )
    return stem, batch


def _late_transfer_surface(
    authority: LateProducerPhysicalAuthority,
) -> FrozenMap:
    rows: dict[tuple[str, str], list[str]] = {}
    batch_positions: dict[tuple[str, str], list[int | None]] = {}
    seen_targets: set[tuple[str, str]] = set()
    for index, group in enumerate(authority.transfer_groups):
        family, batch = _transfer_family_base(
            group.family,
            location=f"late_producer/transfer_groups/{index}/family",
        )
        key = (group.entity, family)
        positions = batch_positions.setdefault(key, [])
        if positions and (batch is None or any(item is None for item in positions)):
            raise USPoolPhysicalAuthorityError(
                "late producer transfer groups ambiguously repeat an unbounded "
                f"family: {group.entity}/{family}"
            )
        positions.append(batch)
        rows.setdefault(key, []).extend(group.targets)
        for target in group.targets:
            target_key = (group.entity, target)
            if target_key in seen_targets:
                raise USPoolPhysicalAuthorityError(
                    "late producer transfer surface repeats target "
                    f"{group.entity}.{target}"
                )
            seen_targets.add(target_key)

    for (entity, family), positions in batch_positions.items():
        bounded = tuple(item for item in positions if item is not None)
        if bounded and bounded != tuple(range(1, len(bounded) + 1)):
            raise USPoolPhysicalAuthorityError(
                "late producer bounded families are not contiguous for "
                f"{entity}/{family}: {bounded}"
            )

    source_order = {
        key: index for index, key in enumerate(authority.source_output_order)
    }

    result: dict[str, dict[str, list[str]]] = {}
    for (entity, family), targets in rows.items():
        ordered_targets = list(targets)
        if all((entity, target) in source_order for target in ordered_targets):
            ordered_targets.sort(key=lambda target: source_order[(entity, target)])
        result.setdefault(entity, {})[family] = ordered_targets
    frozen = freeze_json(result)
    assert isinstance(frozen, FrozenMap)
    return frozen


def _surface_target_keys(surface: FrozenMap, *, location: str) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for entity, families_value in surface.entries:
        families = _map(families_value, location=f"{location}/{entity}")
        for family, targets_value in families.entries:
            targets = _strings(
                targets_value,
                location=f"{location}/{entity}/{family}",
                nonempty=True,
            )
            for target in targets:
                key = (entity, target)
                if key in result:
                    raise USPoolPhysicalAuthorityError(
                        f"{location}: target appears in multiple families: "
                        f"{entity}.{target}"
                    )
                result.add(key)
    return result


def _filter_surface(
    surface: FrozenMap,
    *,
    allowed: set[tuple[str, str]],
    location: str,
) -> FrozenMap:
    result: dict[str, dict[str, list[str]]] = {}
    for entity, families_value in surface.entries:
        families = _map(families_value, location=f"{location}/{entity}")
        for family, targets_value in families.entries:
            targets = _strings(
                targets_value,
                location=f"{location}/{entity}/{family}",
                nonempty=True,
            )
            selected = [target for target in targets if (entity, target) in allowed]
            if selected:
                result.setdefault(entity, {})[family] = selected
    frozen = freeze_json(result)
    assert isinstance(frozen, FrozenMap)
    return frozen


def _authority_component(
    components: FrozenMap,
    name: str,
) -> FrozenMap:
    return _map(
        components.get(name),
        location=f"execution/stacked_authority/components/{name}",
    )


def _assert_component_digest(
    payload: object,
    component: FrozenMap,
    *,
    location: str,
) -> None:
    expected = _string(component.get("sha256"), location=f"{location}/sha256")
    declared = _string(
        component.get("declared_sha256"),
        location=f"{location}/declared_sha256",
    )
    observed = sha256_json(freeze_json(payload))
    if (
        observed != expected
        or observed != declared
        or component.get("digest_matches_declared") is not True
    ):
        raise USPoolPhysicalAuthorityError(
            f"{location}: materialized value differs from compiler digest"
        )


def _compile_terminal(
    plan: USPoolRuntimePlan,
    *,
    pool_code: FrozenMap,
    gap_fill: GapFillPhysicalAuthority,
    primary_qrf: PrimaryQRFPhysicalAuthority,
    late_producers: LateProducerPhysicalAuthority,
    battery: BatteryPhysicalAuthority,
) -> TerminalPhysicalAuthority:
    receipt = plan.execution.stacked_authority
    components = _map(
        receipt.get("components"),
        location="execution/stacked_authority/components",
    )
    component_names = {
        "gap_fill_plan",
        "post_puf_transfer_surface",
        "declared_surface",
        "metric_registry",
        "joint_metric_registry",
        "support_profile",
        "puf_capital_gains_tail_support_contract",
        "late_producer_schedule",
    }
    if set(components) != component_names:
        raise USPoolPhysicalAuthorityError(
            "execution/stacked_authority: component surface changed"
        )

    transfer_surface = _late_transfer_surface(late_producers)
    transfer_keys = _surface_target_keys(
        transfer_surface,
        location="terminal/post_puf_transfer_surface",
    )
    primary_keys = {
        (
            _string(
                output.get("entity"),
                location="terminal/primary_qrf/output/entity",
            ),
            _string(
                output.get("column"),
                location="terminal/primary_qrf/output/column",
            ),
        )
        for output in primary_qrf.node.outputs
        if output.get("coverage_scope") == "puf_clone"
    }
    source_keys = set(late_producers.source_output_order)
    puf_surface = _filter_surface(
        transfer_surface,
        allowed=primary_keys,
        location="terminal/post_puf_puf_producer_surface",
    )
    source_surface = _filter_surface(
        transfer_surface,
        allowed=source_keys,
        location="terminal/post_puf_source_producer_surface",
    )
    puf_keys = _surface_target_keys(
        puf_surface,
        location="terminal/post_puf_puf_producer_surface",
    )
    projected_source_keys = _surface_target_keys(
        source_surface,
        location="terminal/post_puf_source_producer_surface",
    )
    if puf_keys | projected_source_keys != transfer_keys:
        missing = sorted(transfer_keys - puf_keys - projected_source_keys)
        raise USPoolPhysicalAuthorityError(
            "terminal post-PUF targets have no declared producer role: "
            f"{missing}"
        )

    post_component = _authority_component(
        components,
        "post_puf_transfer_surface",
    )
    post_payload = {
        "donor_channel": _string(
            post_component.get("donor_channel"),
            location="stacked_authority/post_puf_transfer_surface/donor_channel",
        ),
        "donor_clone_index": _integer(
            post_component.get("donor_clone_index"),
            location=(
                "stacked_authority/post_puf_transfer_surface/donor_clone_index"
            ),
        ),
        "recipient_selection": _string(
            post_component.get("recipient_selection"),
            location=(
                "stacked_authority/post_puf_transfer_surface/recipient_selection"
            ),
        ),
        "producer_surfaces": {
            "puf_clone": puf_surface,
            "post_clone_source": source_surface,
        },
        "target_families": transfer_surface,
    }
    if _integer(
        post_component.get("target_count"),
        location="stacked_authority/post_puf_transfer_surface/target_count",
        minimum=1,
    ) != len(transfer_keys):
        raise USPoolPhysicalAuthorityError(
            "terminal post-PUF target count differs from compiler receipt"
        )
    if _integer(
        post_component.get("puf_producer_target_count"),
        location=(
            "stacked_authority/post_puf_transfer_surface/"
            "puf_producer_target_count"
        ),
    ) != len(puf_keys):
        raise USPoolPhysicalAuthorityError(
            "terminal PUF-producer count differs from compiler receipt"
        )
    if _integer(
        post_component.get("source_producer_target_count"),
        location=(
            "stacked_authority/post_puf_transfer_surface/"
            "source_producer_target_count"
        ),
    ) != len(projected_source_keys):
        raise USPoolPhysicalAuthorityError(
            "terminal source-producer count differs from compiler receipt"
        )

    tail_roles = tuple(
        _map(
            role.get("tail_support"),
            location=f"spine/support_roles/{index}/tail_support",
        )
        for index, role in enumerate(plan.support_spine.roles)
        if role.get("tail_support") is not None
    )
    tail = _one(tail_roles, location="spine/support_roles tail support")
    tail_contract = _map(
        tail.get("legacy_contract"),
        location="spine/support_roles/tail_support/legacy_contract",
    )
    _same(
        pool_code.get("puf_capital_gains_tail_support_contract"),
        tail_contract,
        location="checkpoint tail-support contract",
    )
    tail_component = _authority_component(
        components,
        "puf_capital_gains_tail_support_contract",
    )
    _same(
        tail_component.get("identity"),
        tail_contract,
        location="stacked-authority tail-support identity",
    )
    late_component = _authority_component(components, "late_producer_schedule")
    _same(
        late_component.get("identity"),
        late_producers.schedule_receipt,
        location="stacked-authority late-producer schedule identity",
    )

    component_payloads: dict[str, object] = {
        "gap_fill_plan": gap_fill.plan_wire(),
        "post_puf_transfer_surface": post_payload,
        "declared_surface": battery.declared_surface,
        "metric_registry": _metric_registry_payload(battery.metric_registry),
        "joint_metric_registry": _joint_metric_registry_payload(
            battery.joint_metric_registry
        ),
        "support_profile": {
            key: battery.support_profile[key]
            for key in ("min_effective_support", "profile_id", "version")
        },
        "puf_capital_gains_tail_support_contract": tail_contract,
        "late_producer_schedule": late_producers.schedule_receipt,
    }
    for name, payload in component_payloads.items():
        _assert_component_digest(
            payload,
            _authority_component(components, name),
            location=f"execution/stacked_authority/components/{name}",
        )

    gap_component = _authority_component(components, "gap_fill_plan")
    if _integer(
        gap_component.get("direction_count"),
        location="stacked_authority/gap_fill_plan/direction_count",
        minimum=1,
    ) != len(gap_fill.directions):
        raise USPoolPhysicalAuthorityError(
            "terminal gap-fill direction count differs from compiler receipt"
        )
    gap_target_count = sum(
        len(targets)
        for direction in gap_fill.directions
        for families_value in direction.target_families.values()
        for targets in _map(
            families_value,
            location=f"terminal/gap_fill/{direction.name}/target_families",
        ).values()
    )
    if _integer(
        gap_component.get("target_count"),
        location="stacked_authority/gap_fill_plan/target_count",
        minimum=1,
    ) != gap_target_count:
        raise USPoolPhysicalAuthorityError(
            "terminal gap-fill target count differs from compiler receipt"
        )
    declared_keys = _surface_target_keys(
        battery.declared_surface,
        location="terminal/declared_surface",
    )
    declared_component = _authority_component(components, "declared_surface")
    if _integer(
        declared_component.get("target_count"),
        location="stacked_authority/declared_surface/target_count",
        minimum=1,
    ) != len(declared_keys):
        raise USPoolPhysicalAuthorityError(
            "terminal declared-surface count differs from compiler receipt"
        )
    if _integer(
        declared_component.get("entity_count"),
        location="stacked_authority/declared_surface/entity_count",
        minimum=1,
    ) != len(battery.declared_surface):
        raise USPoolPhysicalAuthorityError(
            "terminal declared entity count differs from compiler receipt"
        )
    metric_component = _authority_component(components, "metric_registry")
    joint_component = _authority_component(components, "joint_metric_registry")
    if _integer(
        metric_component.get("target_count"),
        location="stacked_authority/metric_registry/target_count",
        minimum=1,
    ) != len(battery.metric_registry):
        raise USPoolPhysicalAuthorityError(
            "terminal metric count differs from compiler receipt"
        )
    if _integer(
        joint_component.get("target_count"),
        location="stacked_authority/joint_metric_registry/target_count",
    ) != len(battery.joint_metric_registry):
        raise USPoolPhysicalAuthorityError(
            "terminal joint-metric count differs from compiler receipt"
        )
    support_component = _authority_component(components, "support_profile")
    for key in ("profile_id", "version", "min_effective_support"):
        if support_component.get(key) != battery.support_profile.get(key):
            raise USPoolPhysicalAuthorityError(
                f"terminal support profile differs from compiler receipt at {key}"
            )
    if _integer(
        late_component.get("producer_count"),
        location="stacked_authority/late_producer_schedule/producer_count",
        minimum=1,
    ) != len(late_producers.producer_order):
        raise USPoolPhysicalAuthorityError(
            "terminal late-producer count differs from compiler receipt"
        )

    bundle_payload = {
        "authority_id": _string(
            receipt.get("authority_id"),
            location="execution/stacked_authority/authority_id",
        ),
        "version": _integer(
            receipt.get("version"),
            location="execution/stacked_authority/version",
            minimum=1,
        ),
        "components": component_payloads,
    }
    bundle_sha256 = sha256_json(freeze_json(bundle_payload))
    receipt_sha256 = _string(
        receipt.get("sha256"),
        location="execution/stacked_authority/sha256",
    )
    declared_sha256 = _string(
        receipt.get("declared_sha256"),
        location="execution/stacked_authority/declared_sha256",
    )
    if bundle_sha256 != receipt_sha256 or bundle_sha256 != declared_sha256:
        raise USPoolPhysicalAuthorityError(
            "terminal stacked authority differs from its compiler bundle digest"
        )
    return TerminalPhysicalAuthority(
        gap_fill_directions=gap_fill.directions,
        post_puf_transfer_surface=transfer_surface,
        post_puf_puf_producer_surface=puf_surface,
        post_puf_source_producer_surface=source_surface,
        declared_surface=battery.declared_surface,
        metric_registry=battery.metric_registry,
        joint_metric_registry=battery.joint_metric_registry,
        support_profile=battery.support_profile,
        puf_capital_gains_tail_support_contract=tail_contract,
        late_producer_schedule=late_producers.schedule_receipt,
        compatibility_receipt=receipt,
    )


def _compile_simulation(
    plan: USPoolRuntimePlan,
    *,
    assembly: AssemblyPhysicalAuthority,
    static: FrozenMap,
    pool_code: FrozenMap,
    seeds: SeedPhysicalAuthority,
) -> SimulationSettings:
    operator_order = _strings(
        pool_code.get("operator_order"),
        location="checkpoint_static_components/pool_code/operator_order",
        nonempty=True,
    )
    execution_order = tuple(operation.id for operation in plan.execution.operations)
    if operator_order != execution_order:
        raise USPoolPhysicalAuthorityError(
            "checkpoint operator order differs from the execution plan"
        )
    sampling = plan.assembly_sampling.runtime_sampling
    fraction = _map(sampling.get("fraction"), location="sampling/fraction")
    exact_count_rule = _string(
        sampling.get("exact_count_rule"),
        location="sampling/exact_count_rule",
    )
    if set(seeds.sampling_channels) != set(assembly.household_mass_shares):
        raise USPoolPhysicalAuthorityError(
            "sampling channels differ from assembly mass-share channels"
        )
    if assembly.mass_anchor_channel != seeds.sampling_channels[0]:
        raise USPoolPhysicalAuthorityError(
            "assembly mass anchor differs from the first sampling channel"
        )
    attachment_fractions: list[float] = []
    for index, role in enumerate(plan.support_spine.roles):
        attachment_value = role.get("attachment")
        if attachment_value is None:
            continue
        attachment = _map(
            attachment_value,
            location=f"spine/support_roles/{index}/attachment",
        )
        attachment_fraction_contract = _map(
            attachment.get("fraction"),
            location=f"spine/support_roles/{index}/attachment/fraction",
        )
        attachment_fractions.append(
            _number(
                attachment_fraction_contract.get("default"),
                location=(
                    f"spine/support_roles/{index}/attachment/fraction/default"
                ),
            )
        )
    attachment_fraction = _one(
        tuple(attachment_fractions),
        location="spine/support_roles attachment fraction",
    )
    if not 0 <= attachment_fraction <= 1:
        raise USPoolPhysicalAuthorityError(
            "spine/support_roles attachment fraction must be in [0, 1]"
        )
    return SimulationSettings(
        target_period=_integer(
            static.get("period"),
            location="checkpoint_static_components/period",
            minimum=1,
        ),
        model_seed=seeds.model_seed,
        sampling_seed=seeds.sampling_seed,
        household_batch_size=_integer(
            pool_code.get("simulation_household_batch_size"),
            location="pool_code/simulation_household_batch_size",
            minimum=1,
        ),
        operator_order=operator_order,
        exact_count_rule=exact_count_rule,
        sample_fraction_default=_number(
            fraction.get("default"), location="sampling/fraction/default"
        ),
        sample_rungs=_rows(
            fraction.get("rungs"),
            location="sampling/fraction/rungs",
            nonempty=True,
        ),
        sampling_channels=seeds.sampling_channels,
        clone_attachment_fraction_default=attachment_fraction,
        household_mass_shares=assembly.household_mass_shares,
        mass_anchor_channel=assembly.mass_anchor_channel,
    )


def compile_us_pool_physical_authority(
    plan: USPoolRuntimePlan,
) -> USPoolPhysicalAuthority:
    """Compile the sole bundle-mode physical authority from a sealed plan."""

    return USPoolPhysicalAuthority.from_runtime_plan(plan)


__all__ = [
    "AssemblyPhysicalAuthority",
    "BatteryGateAuthority",
    "BatteryPhysicalAuthority",
    "GapFillPhysicalAuthority",
    "LateProducerPhysicalAuthority",
    "LateTransferGroup",
    "PhysicalGapFillAbsenceRule",
    "PhysicalGapFillDirection",
    "PrimaryQRFPhysicalAuthority",
    "RemainingStageInputAuthority",
    "RemainingStagePhysicalAuthority",
    "SeedPhysicalAuthority",
    "SimulationSettings",
    "TakeUpPhysicalAuthority",
    "TakeUpProgramAuthority",
    "TakeUpStepAuthority",
    "TerminalPhysicalAuthority",
    "TransferModelParameters",
    "USPoolPhysicalAuthority",
    "USPoolPhysicalAuthorityError",
    "compile_us_pool_physical_authority",
]
