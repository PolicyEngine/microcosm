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
from populace.build.us_runtime.take_up_contract import load_take_up_contract
from populace.frame import Frame

__all__ = [
    "POOL_HOUSEHOLD_MASS_SHARES",
    "POOL_OPERATOR_ORDER",
    "MultispinePoolResult",
    "PoolStageOutput",
    "pool_transfer_target_families",
    "run_multispine_pool_path",
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
    """Return the fixed pool transfer plan checked by the #581 gate.

    The existing ACS QRF declaration remains the base. Every take-up input in
    the checked-in contract is added when that declaration does not already
    own it, because the ACS peer has no measured take-up flags and the terminal
    default agreement registry requires the complete take-up surface.
    """

    plan = {
        entity: {family: tuple(columns) for family, columns in families.items()}
        for entity, families in declared_acs_transfer_target_families().items()
    }
    owned = {
        (entity, column)
        for entity, families in plan.items()
        for columns in families.values()
        for column in columns
    }
    additions: dict[str, list[str]] = {}
    for program in load_take_up_contract().programs:
        key = (program.entity, program.variable)
        if key in owned:
            continue
        additions.setdefault(program.entity, []).append(program.variable)
        owned.add(key)
    for entity, columns in additions.items():
        families = plan.setdefault(entity, {})
        existing = families.get("take_up", ())
        families["take_up"] = (*existing, *sorted(columns))
    return plan


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
