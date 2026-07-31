"""Canonical pre-calibration build path for the assembled US spine pool.

The path is intentionally small and order-bearing:

``assemble -> clone -> impute -> derive -> seed -> simulate -> agreement``.

Assembly owns source provenance. Every later operator receives the whole pool
and must be source-spine blind; PUF-detail routing is clone-index based. The
simulation copy exists only to evaluate formula-owned agreement outputs and is
not the input-only pool returned for H5 publication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.us_runtime.acs_transfer import (
    TargetFamilies,
    declared_acs_transfer_target_families,
)
from populace.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from populace.build.us_runtime.spine_agreement import spine_agreement_gate
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.support_provenance import (
    spine_assembly_receipt,
    spine_provenance_counts,
    validate_assembly_provenance,
)
from populace.build.us_runtime.take_up import with_us_take_up_inputs
from populace.build.us_runtime.take_up_contract import (
    TakeUpProgram,
    load_take_up_contract,
)
from populace.frame import Frame

__all__ = [
    "POOL_HOUSEHOLD_MASS_SHARES",
    "POOL_OPERATOR_ORDER",
    "POOL_RANDOM_SEED",
    "POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE",
    "POOL_TIME_PERIOD",
    "MultispinePoolResult",
    "PoolStageOutput",
    "materialize_multispine_agreement_outputs",
    "pool_transfer_target_families",
    "run_multispine_pool_path",
    "seed_multispine_pool_inputs",
]

POOL_HOUSEHOLD_MASS_SHARES: Mapping[str, float] = {
    "asec": 0.5,
    "acs": 0.5,
}
"""Fixed peer-spine mass shares; calibration remains downstream."""

POOL_OPERATOR_ORDER = (
    "assemble",
    "clone",
    "impute",
    "derive",
    "seed",
    "simulate",
    "agreement",
)
"""The executable pool-build order, including the terminal QA evaluation."""

POOL_RANDOM_SEED = 0
"""Fixed seed shared by pool imputations and seeded input stages."""

POOL_TIME_PERIOD = 2024
"""PolicyEngine period of the 2024 source pool."""

POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE = 5_000
"""Fixed household batch size for terminal formula-output evaluation."""


class _PoolRulesEngine(Protocol):
    def default_values(self, names: list[str]) -> Mapping[str, object]: ...

    def variable_metadata(self, name: str) -> object: ...

    def variables(self) -> list[str]: ...

    def materialize(
        self,
        bundle: Frame,
        variables: list[str],
        period: int,
    ) -> Mapping[str, np.ndarray]: ...


@dataclass(frozen=True)
class PoolStageOutput:
    """One source-blind operator result and its manifest-ready receipt."""

    frame: Frame
    receipt: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError(
                "PoolStageOutput.frame must be a Frame, got "
                f"{type(self.frame).__name__}."
            )
        if not isinstance(self.receipt, Mapping):
            raise TypeError("PoolStageOutput.receipt must be a mapping.")


@dataclass(frozen=True)
class MultispinePoolResult:
    """Input-only pool plus receipts from its terminal agreement evaluation."""

    frame: Frame
    assembly_receipt: Mapping[str, object]
    provenance_counts: Mapping[str, Mapping[str, object]]
    stage_receipts: Mapping[str, Mapping[str, object]]
    agreement_gate: GateResult

    @property
    def simulation_ready(self) -> bool:
        """Whether the unchanged terminal agreement gate passed."""

        return self.agreement_gate.passed


type PoolOperator = Callable[[Frame], PoolStageOutput]
type AgreementGate = Callable[[Frame], GateResult]


def pool_transfer_target_families() -> TargetFamilies:
    """Return the fixed raw-preserving QRF transfer plan.

    The #581 default agreement registry supplements this declaration with the
    complete take-up inventory and formula-owned SSI. Take-up inputs not owned
    by the declared QRF are handled together in the later seed stage, where
    sourced TANF/EITC draws and explicitly disclosed engine defaults remain
    distinguishable in the receipt.
    """

    return {
        entity: {family: tuple(columns) for family, columns in families.items()}
        for entity, families in declared_acs_transfer_target_families().items()
    }


def seed_multispine_pool_inputs(
    frame: Frame,
    *,
    engine: _PoolRulesEngine | None = None,
) -> PoolStageOutput:
    """Seed sourced flags, then disclose and fill unresolved engine defaults.

    TANF and EITC use their checked-in administrative seed mechanisms over the
    whole assembled pool. Existing non-null values survive byte-for-byte.
    Other take-up owners are not fabricated here: any unresolved cells receive
    the installed engine's declared default and the receipt names that fact,
    the contract treatment, and its scope owner/follow-up evidence.
    """

    contract = load_take_up_contract()
    before = _take_up_snapshots(frame, contract.programs)
    seeded = with_us_take_up_inputs(
        frame,
        seed=POOL_RANDOM_SEED,
        time_period=POOL_TIME_PERIOD,
    )
    _assert_take_up_values_preserved(before, seeded)

    rules_engine = engine
    if rules_engine is None:
        from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

        rules_engine = PolicyEngineUSEngine()
    names = [program.variable for program in contract.programs]
    defaults = dict(rules_engine.default_values(names))
    transfer_owned = {
        column
        for families in pool_transfer_target_families().values()
        for columns in families.values()
        for column in columns
    }

    tables = {entity: seeded.table(entity).copy() for entity in seeded.entities}
    programs: dict[str, dict[str, object]] = {}
    for program in contract.programs:
        table = tables[program.entity]
        if program.variable in table:
            values = table[program.variable].copy()
            missing = values.isna()
        else:
            values = pd.Series(pd.NA, index=table.index, dtype="boolean")
            missing = pd.Series(True, index=table.index)

        if program.is_seeded:
            if missing.any():
                raise ValueError(
                    f"Seeded take-up input {program.variable!r} still has "
                    f"{int(missing.sum())} missing row(s)."
                )
            provenance_kind = "administrative_seed_or_preserved_input"
            defaulted_rows = 0
            seeded_rows = int(
                len(table)
                if program.variable not in before
                else before[program.variable][1].isna().sum()
            )
        elif program.variable in transfer_owned:
            if missing.any():
                raise ValueError(
                    f"Transfer-owned take-up input {program.variable!r} still "
                    f"has {int(missing.sum())} missing row(s); refusing to hide "
                    "an incomplete transfer behind an engine default."
                )
            provenance_kind = "transferred_or_preserved_input"
            defaulted_rows = 0
            seeded_rows = 0
        else:
            if program.variable not in defaults:
                raise ValueError(
                    "PolicyEngine exposes no input default for take-up variable "
                    f"{program.variable!r}."
                )
            default = defaults[program.variable]
            if default != program.default:
                raise ValueError(
                    f"Take-up contract default for {program.variable!r} is "
                    f"{program.default!r}, but the installed engine reports "
                    f"{default!r}."
                )
            values.loc[missing] = default
            table[program.variable] = values.astype(bool)
            provenance_kind = "preserved_input_or_disclosed_engine_default"
            defaulted_rows = int(missing.sum())
            seeded_rows = 0

        rate = program.rate
        programs[program.variable] = {
            "entity": program.entity,
            "populace_treatment": program.populace_treatment,
            "provenance_kind": provenance_kind,
            "preserved_nonnull_rows": int(len(table) - defaulted_rows),
            "seeded_rows": seeded_rows,
            "defaulted_rows": defaulted_rows,
            "engine_default": program.default,
            "administrative_source": rate.get("source"),
            "administrative_rate_status": rate.get("status"),
            "scope_owner": program.raw.get("scope_owner"),
            "followup": program.raw.get("followup"),
        }

    result = Frame(
        tables,
        seeded.schema,
        {entity: seeded.weights_for(entity) for entity in seeded.weighted_entities},
        seeded.strata,
        mass_log=seeded.mass_log,
        metadata=seeded.metadata,
    )
    _assert_take_up_values_preserved(before, result)
    return PoolStageOutput(
        result,
        {
            "seed": POOL_RANDOM_SEED,
            "time_period": POOL_TIME_PERIOD,
            "programs": programs,
        },
    )


def materialize_multispine_agreement_outputs(
    frame: Frame,
    *,
    engine: _PoolRulesEngine | None = None,
) -> PoolStageOutput:
    """Materialize SSI in fixed household batches on an ephemeral gate view.

    The returned frame preserves the assembly receipt and adds ``person.ssi``.
    The caller must gate this view and publish :attr:`MultispinePoolResult.frame`
    instead; persisting ``ssi`` would pin a formula-owned output and mask
    reforms.
    """

    if any("ssi" in frame.table(entity) for entity in frame.entities):
        raise ValueError(
            "Multispine agreement simulation refuses a persisted 'ssi' column; "
            "SSI must remain formula-owned and gate-view-only."
        )
    rules_engine = engine
    if rules_engine is None:
        from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

        rules_engine = PolicyEngineUSEngine()

    simulation_frame, default_fills = _simulation_projection(frame, rules_engine)
    household_ids = simulation_frame.table("household")["household_id"].to_numpy()
    person = frame.table("person")
    membership = person["person_household_id"]
    person_ids = person["person_id"]
    if person_ids.duplicated().any():
        raise ValueError("Multispine SSI materialization requires unique person IDs.")

    values_by_person_id = pd.Series(
        np.nan,
        index=pd.Index(person_ids.to_numpy(), name="person_id"),
        dtype=np.float64,
    )
    batch_count = 0
    for low in range(0, len(household_ids), POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE):
        selected_households = household_ids[
            low : low + POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE
        ]
        person_mask = membership.isin(selected_households).to_numpy()
        selected = simulation_frame.select(person_mask)
        materialized = np.asarray(
            rules_engine.materialize(
                selected,
                ["ssi"],
                POOL_TIME_PERIOD,
            )["ssi"],
            dtype=np.float64,
        )
        selected_ids = selected.table("person")["person_id"].to_numpy()
        if materialized.shape != (len(selected_ids),):
            raise ValueError(
                "Materialized SSI does not align with the selected person table."
            )
        values_by_person_id.loc[selected_ids] = materialized
        batch_count += 1

    if values_by_person_id.isna().any():
        raise ValueError(
            "Batched SSI materialization did not cover every person exactly once."
        )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["ssi"] = values_by_person_id.reindex(person_ids).to_numpy()
    evaluation = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return PoolStageOutput(
        evaluation,
        {
            "formula_outputs": {
                "ssi": {
                    "entity": "person",
                    "period": POOL_TIME_PERIOD,
                    "rows": int(len(person)),
                }
            },
            "household_batch_size": POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
            "batches": batch_count,
            "simulation_projection_default_fills": default_fills,
            "persisted_to_pool": False,
        },
    )


def _simulation_projection(
    frame: Frame,
    engine: _PoolRulesEngine,
) -> tuple[Frame, dict[str, dict[str, object]]]:
    """Fill nullable engine inputs only on the disposable simulation copy."""

    variables_method = getattr(engine, "variables", None)
    metadata_method = getattr(engine, "variable_metadata", None)
    if not callable(variables_method) or not callable(metadata_method):
        return frame, {}

    input_names = list(variables_method())
    defaults = dict(engine.default_values(input_names))
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    fills: dict[str, dict[str, object]] = {}
    for name in input_names:
        metadata = metadata_method(name)
        entity = getattr(metadata, "entity", None)
        if entity not in tables or name not in tables[entity]:
            continue
        missing = tables[entity][name].isna()
        if not missing.any():
            continue
        if name not in defaults:
            raise ValueError(
                f"SSI simulation projection cannot resolve {int(missing.sum())} "
                f"missing value(s) in engine input {entity}.{name}; the engine "
                "declares no default."
            )
        tables[entity].loc[missing, name] = defaults[name]
        fills[name] = {
            "entity": entity,
            "rows": int(missing.sum()),
            "value": defaults[name],
            "persisted_to_pool": False,
        }
    if not fills:
        return frame, {}
    return (
        Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        ),
        fills,
    )


def _take_up_snapshots(
    frame: Frame,
    programs: tuple[TakeUpProgram, ...],
) -> dict[str, tuple[str, pd.Series, pd.Series]]:
    snapshots: dict[str, tuple[str, pd.Series, pd.Series]] = {}
    for program in programs:
        table = frame.table(program.entity)
        if program.variable not in table:
            continue
        values = table[program.variable].copy(deep=True)
        snapshots[program.variable] = (
            program.entity,
            values,
            values.notna(),
        )
    return snapshots


def _assert_take_up_values_preserved(
    snapshots: Mapping[str, tuple[str, pd.Series, pd.Series]],
    frame: Frame,
) -> None:
    for variable, (entity, before, observed) in snapshots.items():
        after = frame.table(entity)[variable]
        before_values = before.loc[observed].to_numpy(dtype=object)
        after_values = after.loc[observed].to_numpy(dtype=object)
        if not np.array_equal(before_values, after_values):
            raise ValueError(
                f"Pool take-up stage would overwrite non-null input "
                f"{entity}.{variable}; measured/source-owned values must remain "
                "untouched."
            )


def run_multispine_pool_path(
    asec: Frame,
    acs: Frame,
    *,
    impute: PoolOperator,
    derive: PoolOperator,
    seed: PoolOperator,
    simulate: PoolOperator,
    agreement_gate: AgreementGate | None = None,
) -> MultispinePoolResult:
    """Run the fixed assembly-to-agreement path over two peer source frames.

    ``impute``, ``derive``, and ``seed`` each receive the entire cloned pool.
    They have no source label argument and are checked at their output boundary
    against the immutable assembly receipt. ``simulate`` returns a temporary
    evaluation frame: formula-owned outputs on that copy are visible to the
    terminal gate but never enter :attr:`MultispinePoolResult.frame`.

    ``agreement_gate`` is an injection seam for small synthetic tests only.
    Production callers omit it, which invokes
    :func:`~populace.build.us_runtime.spine_agreement.spine_agreement_gate`
    with its fixed registry and tolerances.
    """

    operators = {
        "impute": impute,
        "derive": derive,
        "seed": seed,
        "simulate": simulate,
    }
    invalid = [name for name, operator in operators.items() if not callable(operator)]
    if invalid:
        raise TypeError(f"Pool operator(s) are not callable: {invalid}.")

    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares=POOL_HOUSEHOLD_MASS_SHARES,
        mass_anchor_channel="asec",
    )
    assembly_receipt = spine_assembly_receipt(
        assembled,
        boundary="multispine pool assembly",
    )

    current = clone_us_frame_for_puf_support(assembled)
    validate_assembly_provenance(
        current,
        boundary="multispine pool clone output",
    )

    receipts: dict[str, Mapping[str, object]] = {}
    for stage_name in ("impute", "derive", "seed"):
        outcome = operators[stage_name](current)
        if not isinstance(outcome, PoolStageOutput):
            raise TypeError(
                f"Pool {stage_name} operator must return PoolStageOutput, got "
                f"{type(outcome).__name__}."
            )
        current = outcome.frame
        validate_assembly_provenance(
            current,
            boundary=f"multispine pool {stage_name} output",
        )
        receipts[stage_name] = dict(outcome.receipt)

    counts = spine_provenance_counts(
        current,
        boundary="multispine pool pre-agreement output",
    )
    simulated = operators["simulate"](current)
    if not isinstance(simulated, PoolStageOutput):
        raise TypeError(
            "Pool simulate operator must return PoolStageOutput, got "
            f"{type(simulated).__name__}."
        )
    validate_assembly_provenance(
        simulated.frame,
        boundary="multispine pool simulation output",
    )
    receipts["simulate"] = dict(simulated.receipt)

    gate_operator = spine_agreement_gate if agreement_gate is None else agreement_gate
    agreement = gate_operator(simulated.frame)
    if not isinstance(agreement, GateResult):
        raise TypeError(
            "Pool agreement operator must return GateResult, got "
            f"{type(agreement).__name__}."
        )
    return MultispinePoolResult(
        frame=current,
        assembly_receipt=assembly_receipt,
        provenance_counts=counts,
        stage_receipts=receipts,
        agreement_gate=agreement,
    )
