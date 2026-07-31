"""Canonical pre-calibration build path for the assembled US spine pool.

The path is intentionally small and order-bearing:

``assemble -> clone -> impute -> derive -> seed -> simulate -> agreement``.

Assembly owns source provenance. The clone stage first prepares source-derived
values that must exist before row expansion, then physically clones the pool.
Every population operator remains source-spine blind; PUF-detail routing is
clone-index based. The simulation copy exists only to evaluate formula-owned
agreement outputs and is not the input-only pool returned for H5 publication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.us_runtime.acs_transfer import (
    ACS_NATIVE_PERSON_INPUTS,
    TargetFamilies,
    declared_acs_transfer_target_families,
    derive_acs_schedule_d_capital_gain_distributions,
)
from populace.build.us_runtime.adult_care import with_us_adult_care_inputs
from populace.build.us_runtime.child_support import with_us_child_support_inputs
from populace.build.us_runtime.childcare import with_us_childcare_inputs
from populace.build.us_runtime.cps_carried import derive_us_cps_carried_inputs
from populace.build.us_runtime.disability_benefits import (
    with_us_disability_benefits,
)
from populace.build.us_runtime.education_inputs import with_us_education_inputs
from populace.build.us_runtime.eligibility_inputs import (
    with_us_eligibility_inputs,
)
from populace.build.us_runtime.energy_subsidy import (
    with_us_energy_subsidy_input,
)
from populace.build.us_runtime.housing_inputs import (
    impute_us_housing_assistance_to_puf_support,
    with_us_housing_inputs,
)
from populace.build.us_runtime.immigration import with_us_immigration_inputs
from populace.build.us_runtime.medicare_take_up import (
    with_us_medicare_take_up_input,
)
from populace.build.us_runtime.operator_boundary import (
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from populace.build.us_runtime.pregnancy import with_us_pregnancy_inputs
from populace.build.us_runtime.prior_year_income import (
    with_us_prior_year_income_inputs,
)
from populace.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from populace.build.us_runtime.qbi_inputs import (
    US_QBI_OUTPUT_COLUMNS,
    with_us_qbi_input_reconciliation,
)
from populace.build.us_runtime.relationship_inputs import (
    with_us_relationship_inputs,
)
from populace.build.us_runtime.retirement_contributions import (
    with_us_retirement_contribution_inputs,
)
from populace.build.us_runtime.retirement_distributions import (
    with_us_retirement_distribution_inputs,
)
from populace.build.us_runtime.spine_agreement import (
    default_spine_agreement_registry,
    spine_agreement_gate,
)
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.support_provenance import (
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_assembly_receipt,
    spine_provenance_counts,
    support_clone_index_column,
    validate_assembly_provenance,
    without_support_role_metadata,
)
from populace.build.us_runtime.take_up import with_us_take_up_inputs
from populace.build.us_runtime.take_up_contract import (
    TakeUpProgram,
    load_take_up_contract,
)
from populace.build.us_runtime.weeks_unemployed import with_us_weeks_unemployed
from populace.build.us_runtime.wic_claim import with_us_wic_claim_input
from populace.build.us_runtime.workers_compensation import (
    with_us_workers_compensation,
)
from populace.frame import Frame

__all__ = [
    "POOL_HOUSEHOLD_MASS_SHARES",
    "POOL_OPERATOR_ORDER",
    "POOL_RANDOM_SEED",
    "POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE",
    "POOL_POST_CLONE_SOURCE_OPERATOR_ORDER",
    "POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER",
    "POOL_SOURCE_OPERATOR_CONTRACTS",
    "POOL_SOURCE_OPERATOR_ORDER",
    "POOL_SPINE_AGREEMENT_REGISTRY",
    "POOL_TIME_PERIOD",
    "MultispinePoolResult",
    "PoolStageOutput",
    "SourceOperatorContract",
    "complete_multispine_source_inputs",
    "derive_multispine_pool_inputs",
    "materialize_multispine_agreement_outputs",
    "pool_transfer_target_families",
    "prepare_multispine_puf_predictors",
    "prepare_multispine_source_inputs_for_clone",
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

POOL_SOURCE_OPERATOR_ORDER = (
    "derive_us_cps_carried_inputs",
    "with_us_prior_year_income_inputs",
    "with_us_relationship_inputs",
    "with_us_medicare_take_up_input",
    "with_us_housing_inputs",
    "with_us_eligibility_inputs",
    "with_us_pregnancy_inputs",
    "with_us_wic_claim_input",
    "impute_us_housing_assistance_to_puf_support",
    "with_us_child_support_inputs",
    "with_us_disability_benefits",
    "with_us_workers_compensation",
    "with_us_weeks_unemployed",
    "with_us_childcare_inputs",
    "with_us_adult_care_inputs",
    "with_us_energy_subsidy_input",
    "with_us_retirement_contribution_inputs",
    "with_us_retirement_distribution_inputs",
    "with_us_immigration_inputs",
    "with_us_education_inputs",
)
"""Logical source-input ownership inventory in legacy relative order.

Executable placement is declared separately because prior-year income has a
pre-clone derivation and a post-clone PUF-imputation pass.
"""

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
type SourceFrameOperator = Callable[[Frame], Frame]


@dataclass(frozen=True)
class SourceOperatorContract:
    """Clone-phase placement and the mechanism that requires it."""

    family: str
    phases: tuple[str, ...]
    mechanism: str


_PRE_CLONE_PHASE = "pre_clone"
_POST_CLONE_PHASE = "post_clone"

POOL_SOURCE_OPERATOR_CONTRACTS: Mapping[str, SourceOperatorContract] = {
    "derive_us_cps_carried_inputs": SourceOperatorContract(
        "cps_carried",
        (_PRE_CLONE_PHASE,),
        "rowwise measured mappings needed by the pre-clone rent predictors",
    ),
    "with_us_prior_year_income_inputs": SourceOperatorContract(
        "prior_year_income",
        (_PRE_CLONE_PHASE, _POST_CLONE_PHASE),
        "unique adjacent-year source join before cloning, then role-aware PUF QRF",
    ),
    "with_us_relationship_inputs": SourceOperatorContract(
        "relationship_inputs",
        (_PRE_CLONE_PHASE,),
        "household-head input needed by the pre-clone rent recipient",
    ),
    "with_us_medicare_take_up_input": SourceOperatorContract(
        "medicare_take_up",
        (_POST_CLONE_PHASE,),
        "rowwise source carry with clone-role-aware completion",
    ),
    "with_us_housing_inputs": SourceOperatorContract(
        "housing_inputs",
        (_PRE_CLONE_PHASE,),
        "rent must be drawn once per source household and cloned unchanged",
    ),
    "with_us_eligibility_inputs": SourceOperatorContract(
        "eligibility_inputs",
        (_PRE_CLONE_PHASE,),
        "raw household parent pointers must be counted before rows duplicate",
    ),
    "with_us_pregnancy_inputs": SourceOperatorContract(
        "pregnancy",
        (_POST_CLONE_PHASE,),
        "stable source-identity draws are shared across support clones",
    ),
    "with_us_wic_claim_input": SourceOperatorContract(
        "wic_claim",
        (_POST_CLONE_PHASE,),
        "remapped family grouping and stable source-identity draws are clone-safe",
    ),
    "impute_us_housing_assistance_to_puf_support": SourceOperatorContract(
        "housing_assistance",
        (_POST_CLONE_PHASE,),
        "PUF-only assistance replacement requires clone roles",
    ),
    "with_us_child_support_inputs": SourceOperatorContract(
        "child_support",
        (_POST_CLONE_PHASE,),
        "role-aware QRF replacement uses remapped structural IDs",
    ),
    "with_us_disability_benefits": SourceOperatorContract(
        "disability_benefits",
        (_POST_CLONE_PHASE,),
        "role-aware QRF replacement uses remapped structural IDs",
    ),
    "with_us_workers_compensation": SourceOperatorContract(
        "workers_compensation",
        (_POST_CLONE_PHASE,),
        "role-aware QRF replacement uses remapped structural IDs",
    ),
    "with_us_weeks_unemployed": SourceOperatorContract(
        "weeks_unemployed",
        (_POST_CLONE_PHASE,),
        "role-aware QRF replacement uses remapped structural IDs",
    ),
    "with_us_childcare_inputs": SourceOperatorContract(
        "childcare",
        (_POST_CLONE_PHASE,),
        "role-aware QRF replacement uses remapped SPM-unit IDs",
    ),
    "with_us_adult_care_inputs": SourceOperatorContract(
        "adult_care",
        (_POST_CLONE_PHASE,),
        "clone-local unit imputation explicitly requires support roles",
    ),
    "with_us_energy_subsidy_input": SourceOperatorContract(
        "energy_subsidy",
        (_POST_CLONE_PHASE,),
        "role-aware QRF replacement uses remapped SPM-unit IDs",
    ),
    "with_us_retirement_contribution_inputs": SourceOperatorContract(
        "retirement_contributions",
        (_POST_CLONE_PHASE,),
        "role-aware QRF replacement uses remapped structural IDs",
    ),
    "with_us_retirement_distribution_inputs": SourceOperatorContract(
        "retirement_distributions",
        (_POST_CLONE_PHASE,),
        "forced PUF imputation explicitly requires support roles",
    ),
    "with_us_immigration_inputs": SourceOperatorContract(
        "immigration",
        (_POST_CLONE_PHASE,),
        "source-keyed draws preserve equality across support clones",
    ),
    "with_us_education_inputs": SourceOperatorContract(
        "education_inputs",
        (_POST_CLONE_PHASE,),
        "deterministic rowwise derivation follows PUF tuition imputation",
    ),
}

POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER = tuple(
    name
    for name in POOL_SOURCE_OPERATOR_ORDER
    if _PRE_CLONE_PHASE in POOL_SOURCE_OPERATOR_CONTRACTS[name].phases
)
"""Source operations owned by the clone stage before physical expansion."""

POOL_POST_CLONE_SOURCE_OPERATOR_ORDER = tuple(
    name
    for name in POOL_SOURCE_OPERATOR_ORDER
    if _POST_CLONE_PHASE in POOL_SOURCE_OPERATOR_CONTRACTS[name].phases
)
"""Source operations safe or required after physical support expansion."""

_CPS_SOURCE_EVIDENCE_COLUMN = "PERIDNUM"
_SOURCE_OPERATOR_FAMILIES: Mapping[str, str] = {
    name: contract.family
    for name, contract in POOL_SOURCE_OPERATOR_CONTRACTS.items()
}
_FORMULA_OWNED_SOURCE_OUTPUTS: Mapping[str, frozenset[str]] = {
    "person": frozenset({"employment_income_last_year"}),
}
_POOL_NATIVE_COMPLETE_OUTPUTS: Mapping[str, frozenset[str]] = {
    "person": frozenset(
        {
            *ACS_NATIVE_PERSON_INPUTS,
            "age",
            "is_female",
            "is_household_head",
        }
    ),
    "household": frozenset({"tenure_type"}),
    "spm_unit": frozenset({"spm_unit_tenure_type"}),
}


def pool_transfer_target_families() -> TargetFamilies:
    """Return the fixed pool-only raw-preserving QRF transfer plan.

    The legacy declaration remains unchanged. This pool-local copy adds every
    persisted historical source-operator output that ACS does not map natively
    and the legacy plan does not already own. A target appears in exactly one
    family, so transfer provenance stays unambiguous. Formula-owned outputs are
    never transfer targets.

    The #581 agreement registry supplements this plan with the complete take-up
    inventory and formula-owned SSI. Take-up inputs not owned by QRF remain in
    the later seed stage, where sourced draws and disclosed engine defaults
    remain distinguishable in the receipt.
    """

    plan: dict[str, dict[str, tuple[str, ...]]] = {
        entity: {family: tuple(columns) for family, columns in families.items()}
        for entity, families in declared_acs_transfer_target_families().items()
    }
    declared = {
        column
        for families in plan.values()
        for columns in families.values()
        for column in columns
    }
    for operator_name in POOL_SOURCE_OPERATOR_ORDER:
        family = _SOURCE_OPERATOR_FAMILIES[operator_name]
        additions: dict[str, tuple[str, ...]] = {}
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[family].items():
            excluded = (
                declared
                | set(_POOL_NATIVE_COMPLETE_OUTPUTS.get(entity, ()))
                | set(_FORMULA_OWNED_SOURCE_OUTPUTS.get(entity, ()))
            )
            targets = tuple(sorted(set(columns) - excluded))
            if targets:
                additions[entity] = targets
                declared.update(targets)
        for entity, targets in additions.items():
            plan.setdefault(entity, {})[f"source_operator_{family}"] = targets
    return plan


POOL_SPINE_AGREEMENT_REGISTRY = default_spine_agreement_registry(
    pool_transfer_target_families()
)
"""Immutable pool charter: transfers, derived leaves, take-up, and SSI."""


def prepare_multispine_puf_predictors(frame: Frame) -> PoolStageOutput:
    """Derive CPS-carried primary-QRF predictors after assembly, before cloning.

    ``PERIDNUM`` is raw CPS evidence and is absent from the harmonized ACS
    source. Only rows carrying that raw evidence enter the historical
    derivation. Explicit output families are merged back by structural entity
    ID, leaving unavailable peer-spine cells nullable and preserving native
    non-null cells.
    """

    return _run_source_operator_chain(
        frame,
        phase=_PRE_CLONE_PHASE,
        operator_names=("derive_us_cps_carried_inputs",),
        operators={
            "derive_us_cps_carried_inputs": derive_us_cps_carried_inputs,
        },
    )


def prepare_multispine_source_inputs_for_clone(
    frame: Frame,
    *,
    acs_rent_donor: pd.DataFrame,
) -> PoolStageOutput:
    """Prepare source-derived values whose grain would be corrupted by cloning.

    This is the source-blind preparation subphase of the coarse ``clone``
    stage. It runs only after assembly and on a CPS-evidence projection with
    support-role metadata removed. Physical cloning then copies the outputs,
    including the transient prior-year wage target needed by the later PUF QRF.
    """

    operators: Mapping[str, SourceFrameOperator] = {
        "derive_us_cps_carried_inputs": derive_us_cps_carried_inputs,
        "with_us_prior_year_income_inputs": lambda current: (
            with_us_prior_year_income_inputs(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
            )
        ),
        "with_us_relationship_inputs": lambda current: with_us_relationship_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_housing_inputs": lambda current: with_us_housing_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            acs_rent_donor=acs_rent_donor,
        ),
        "with_us_eligibility_inputs": lambda current: with_us_eligibility_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
    }
    return _run_source_operator_chain(
        frame,
        phase=_PRE_CLONE_PHASE,
        operator_names=POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
        operators=operators,
    )


def complete_multispine_source_inputs(
    frame: Frame,
) -> PoolStageOutput:
    """Run clone-safe or clone-required source work after primary imputation.

    The function is intentionally fixed-seed/fixed-period. Each operator runs
    over the CPS-evidenced portion of the already assembled and cloned frame;
    its declared output family alone is merged into the whole pool. This
    retains ACS native measurements, leaves unavailable cells null for the
    subsequent declared transfer, and keeps assembly metadata untouched.
    """

    operators: Mapping[str, SourceFrameOperator] = {
        "with_us_prior_year_income_inputs": lambda current: (
            with_us_prior_year_income_inputs(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
            )
        ),
        "with_us_medicare_take_up_input": lambda current: (
            with_us_medicare_take_up_input(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
            )
        ),
        "with_us_pregnancy_inputs": lambda current: with_us_pregnancy_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_wic_claim_input": lambda current: with_us_wic_claim_input(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "impute_us_housing_assistance_to_puf_support": lambda current: (
            impute_us_housing_assistance_to_puf_support(
                current,
                seed=POOL_RANDOM_SEED,
            )
        ),
        "with_us_child_support_inputs": lambda current: with_us_child_support_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_disability_benefits": lambda current: with_us_disability_benefits(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_workers_compensation": lambda current: (
            with_us_workers_compensation(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
            )
        ),
        "with_us_weeks_unemployed": lambda current: with_us_weeks_unemployed(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_childcare_inputs": lambda current: with_us_childcare_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_adult_care_inputs": lambda current: with_us_adult_care_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_energy_subsidy_input": lambda current: with_us_energy_subsidy_input(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_retirement_contribution_inputs": lambda current: (
            with_us_retirement_contribution_inputs(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
            )
        ),
        "with_us_retirement_distribution_inputs": lambda current: (
            with_us_retirement_distribution_inputs(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
                force_puf_imputation=True,
            )
        ),
        "with_us_immigration_inputs": lambda current: with_us_immigration_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
        "with_us_education_inputs": lambda current: with_us_education_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
        ),
    }
    completed = _run_source_operator_chain(
        frame,
        phase=_POST_CLONE_PHASE,
        operator_names=POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
        operators=operators,
    )
    _assert_formula_owned_source_outputs_absent(completed.frame)
    return completed


def _run_source_operator_chain(
    frame: Frame,
    *,
    phase: str,
    operator_names: tuple[str, ...],
    operators: Mapping[str, SourceFrameOperator],
    output_families: Mapping[
        str,
        Mapping[str, frozenset[str]],
    ] = PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
) -> PoolStageOutput:
    """Run an injectable source-available chain and merge declared outputs."""

    if not isinstance(frame, Frame):
        raise TypeError(
            "Multispine source operators require a Frame, got "
            f"{type(frame).__name__}."
        )
    if phase not in {_PRE_CLONE_PHASE, _POST_CLONE_PHASE}:
        raise ValueError(f"Unknown multispine source-operator phase {phase!r}.")
    expected_operators = set(operator_names)
    if set(operators) != expected_operators:
        raise ValueError(
            "Multispine source operator mapping must exactly match the requested "
            f"order; missing={sorted(expected_operators - set(operators))}, "
            f"unexpected={sorted(set(operators) - expected_operators)}."
        )
    invalid = [name for name, operator in operators.items() if not callable(operator)]
    if invalid:
        raise TypeError(f"Multispine source operator(s) are not callable: {invalid}.")

    misplaced = [
        name
        for name in operator_names
        if name not in POOL_SOURCE_OPERATOR_CONTRACTS
        or phase not in POOL_SOURCE_OPERATOR_CONTRACTS[name].phases
    ]
    if misplaced:
        raise ValueError(
            f"Multispine source operator(s) are not declared for {phase}: "
            f"{misplaced}."
        )

    _assert_source_operator_boundary(frame, phase=phase)
    current = frame
    receipts: list[dict[str, object]] = []
    for order_index, operator_name in enumerate(operator_names):
        family = _SOURCE_OPERATOR_FAMILIES.get(operator_name, operator_name)
        if family not in output_families:
            raise ValueError(
                f"Multispine source operator {operator_name!r} has no declared "
                f"output family {family!r}."
            )
        declared_outputs = (
            dict(output_families[family])
            if phase == _PRE_CLONE_PHASE
            else _persisted_source_outputs(output_families[family])
        )
        available_mask = _cps_source_evidence_mask(current, phase=phase)
        available = _source_available_projection(
            current,
            available_mask,
            phase=phase,
        )
        available = _without_unavailable_output_columns(
            available,
            declared_outputs,
        )
        before_rows = _frame_row_counts(current)
        available_rows = _frame_row_counts(available)
        outcome = operators[operator_name](available)
        if not isinstance(outcome, Frame):
            raise TypeError(
                f"Multispine source operator {operator_name!r} must return Frame, "
                f"got {type(outcome).__name__}."
            )
        output_rows = _frame_row_counts(outcome)
        if output_rows != available_rows:
            raise ValueError(
                f"Multispine source operator {operator_name!r} changed entity row "
                f"counts: input={available_rows}, output={output_rows}."
            )
        _assert_source_operator_structure(
            available,
            outcome,
            operator_name=operator_name,
        )
        current, merged_rows = _merge_source_operator_outputs(
            current,
            outcome,
            declared_outputs,
            operator_name=operator_name,
        )
        formula_owned_removed: dict[str, list[str]] = {}
        if phase == _POST_CLONE_PHASE:
            formula_owned = {
                entity: frozenset(
                    set(columns)
                    & set(_FORMULA_OWNED_SOURCE_OUTPUTS.get(entity, ()))
                )
                for entity, columns in output_families[family].items()
            }
            current, formula_owned_removed = _drop_source_output_columns(
                current,
                formula_owned,
            )
        after_rows = _frame_row_counts(current)
        if after_rows != before_rows:
            raise AssertionError(
                f"Multispine source output merge changed pool row counts at "
                f"{operator_name!r}: input={before_rows}, output={after_rows}."
            )
        receipts.append(
            {
                "order_index": order_index,
                "operator": operator_name,
                "family": family,
                "phase": phase,
                "pool_input_rows": before_rows,
                "cps_available_rows": available_rows,
                "operator_output_rows": output_rows,
                "merged_rows": merged_rows,
                "operator_projection": {
                    "selection": _CPS_SOURCE_EVIDENCE_COLUMN,
                    "lineage_state_persisted": False,
                    "support_role_metadata_exposed": phase == _POST_CLONE_PHASE,
                },
                "output_columns": {
                    entity: sorted(columns)
                    for entity, columns in declared_outputs.items()
                    if columns
                },
                "formula_owned_outputs_removed": formula_owned_removed,
            }
        )
    return PoolStageOutput(
        current,
        {
            "phase": phase,
            "operator_order": list(operator_names),
            "cps_source_evidence": {
                "column": _CPS_SOURCE_EVIDENCE_COLUMN,
                "person_rows": int(
                    _cps_source_evidence_mask(frame, phase=phase).sum()
                ),
            },
            "transient_outputs_carried_through_clone": (
                _transient_source_outputs(operator_names, output_families)
                if phase == _PRE_CLONE_PHASE
                else {}
            ),
            "suboperators": receipts,
        },
    )


def _assert_source_operator_boundary(frame: Frame, *, phase: str) -> None:
    manifest = frame.metadata.get(SPINE_ASSEMBLY_MANIFEST_KEY)
    if not isinstance(manifest, Mapping):
        raise ValueError(
            "Multispine source operators require the immutable spine assembly "
            "manifest before any source derivation."
        )
    person = frame.table(frame.schema.person_entity)
    clone_column = support_clone_index_column(frame.schema.person_entity)
    if clone_column not in person:
        raise ValueError(
            "Multispine source operators require post-assembly clone provenance; "
            f"missing {clone_column!r}."
        )
    clone_index = pd.to_numeric(person[clone_column], errors="coerce")
    if clone_index.isna().any():
        raise ValueError(
            f"Multispine source clone provenance {clone_column!r} must be integral."
        )
    clone_values = clone_index.to_numpy(dtype=np.float64)
    invalid_numeric = (clone_values < 0.0).any() or not np.equal(
        clone_values, np.floor(clone_values)
    ).all()
    pre_clone_invalid = phase == _PRE_CLONE_PHASE and not np.all(
        clone_values == 0.0
    )
    post_clone_invalid = phase == _POST_CLONE_PHASE and (
        not np.any(clone_values == 0.0) or not np.any(clone_values > 0.0)
    )
    if invalid_numeric or pre_clone_invalid or post_clone_invalid:
        raise ValueError(
            f"Multispine {phase} source operators received incompatible clone "
            "provenance; pre-clone requires only index 0, while post-clone "
            "requires both native and positive clone indices."
        )
    _cps_source_evidence_mask(frame, phase=phase)


def _cps_source_evidence_mask(frame: Frame, *, phase: str) -> pd.Series:
    """Select CPS lineage only from a raw column unavailable on ACS."""

    person = frame.table(frame.schema.person_entity)
    if _CPS_SOURCE_EVIDENCE_COLUMN not in person:
        raise ValueError(
            "Multispine source operators require raw CPS evidence column "
            f"{_CPS_SOURCE_EVIDENCE_COLUMN!r}."
        )
    evidence = person[_CPS_SOURCE_EVIDENCE_COLUMN]
    available = evidence.notna()
    if pd.api.types.is_string_dtype(evidence.dtype) or evidence.dtype == object:
        available &= evidence.astype("string").str.strip().ne("").fillna(False)
    if not available.any():
        raise ValueError(
            "Multispine source operators found no CPS-evidenced person rows in "
            f"{_CPS_SOURCE_EVIDENCE_COLUMN!r}."
        )
    clone_column = support_clone_index_column(frame.schema.person_entity)
    clone_index = pd.to_numeric(person[clone_column], errors="coerce")
    evidenced_clones = set(clone_index.loc[available].astype(int).tolist())
    invalid_evidence = (
        evidenced_clones != {0}
        if phase == _PRE_CLONE_PHASE
        else 0 not in evidenced_clones
        or not any(index > 0 for index in evidenced_clones)
    )
    if invalid_evidence:
        raise ValueError(
            f"Raw CPS evidence is incompatible with the {phase} source-operator "
            "boundary."
        )
    return available.astype(bool)


def _source_available_projection(
    frame: Frame,
    person_mask: pd.Series,
    *,
    phase: str,
) -> Frame:
    """Build an ephemeral CPS-only kernel input without a false pool receipt.

    The public source-chain boundary receives and validates the fully assembled
    clone pool. Historical source operators cannot safely consume rows lacking
    their raw CPS inputs, so their internal kernel runs on this structural
    projection. Assembly metadata and mass history describe the full pool and
    therefore must not be attached to the subset. Only declared outputs are
    merged back into the still-receipted full pool.
    """

    selected = frame.select(person_mask)
    tables = {
        entity: (
            without_support_role_metadata(selected.table(entity), entity=entity)
            if phase == _PRE_CLONE_PHASE
            else selected.table(entity).copy()
        )
        for entity in selected.entities
    }
    return Frame(
        tables,
        selected.schema,
        {
            entity: selected.weights_for(entity)
            for entity in selected.weighted_entities
        },
        selected.strata,
    )


def _drop_source_output_columns(
    frame: Frame,
    outputs: Mapping[str, frozenset[str]],
) -> tuple[Frame, dict[str, list[str]]]:
    """Drop transient source outputs from the whole pool after their consumer."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    removed: dict[str, list[str]] = {}
    for entity, columns in outputs.items():
        if entity not in tables:
            continue
        present = sorted(set(columns) & set(tables[entity].columns))
        if present:
            tables[entity] = tables[entity].drop(columns=present)
            removed[entity] = present
    if not removed:
        return frame, removed
    return (
        Frame(
            tables,
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        ),
        removed,
    )


def _transient_source_outputs(
    operator_names: tuple[str, ...],
    output_families: Mapping[str, Mapping[str, frozenset[str]]],
) -> dict[str, list[str]]:
    transients: dict[str, set[str]] = {}
    for operator_name in operator_names:
        family = _SOURCE_OPERATOR_FAMILIES[operator_name]
        for entity, columns in output_families[family].items():
            formula_owned = set(columns) & set(
                _FORMULA_OWNED_SOURCE_OUTPUTS.get(entity, ())
            )
            if formula_owned:
                transients.setdefault(entity, set()).update(formula_owned)
    return {entity: sorted(columns) for entity, columns in transients.items()}


def _assert_formula_owned_source_outputs_absent(frame: Frame) -> None:
    remaining = {
        entity: sorted(set(columns) & set(frame.table(entity).columns))
        for entity, columns in _FORMULA_OWNED_SOURCE_OUTPUTS.items()
        if entity in frame.entities
        and set(columns).intersection(frame.table(entity).columns)
    }
    if remaining:
        raise ValueError(
            "Completed multispine source inputs retain formula-owned transient "
            f"output(s): {remaining}."
        )


def _persisted_source_outputs(
    outputs: Mapping[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    return {
        entity: frozenset(
            set(columns) - set(_FORMULA_OWNED_SOURCE_OUTPUTS.get(entity, ()))
        )
        for entity, columns in outputs.items()
    }


def _assert_source_operator_structure(
    before: Frame,
    after: Frame,
    *,
    operator_name: str,
) -> None:
    if after.metadata != before.metadata or after.mass_log != before.mass_log:
        raise ValueError(
            f"Multispine source operator {operator_name!r} changed immutable "
            "assembly metadata or mass history."
        )
    for entity in before.entities:
        entity_id = before.schema.entity_id_column(entity)
        before_ids = before.table(entity)[entity_id]
        after_ids = after.table(entity)[entity_id]
        if (
            after_ids.duplicated().any()
            or set(after_ids.tolist()) != set(before_ids.tolist())
        ):
            raise ValueError(
                f"Multispine source operator {operator_name!r} changed structural "
                f"{entity_id!r} values."
            )


def _without_unavailable_output_columns(
    frame: Frame,
    outputs: Mapping[str, frozenset[str]],
) -> Frame:
    """Remove union-created all-null outputs before an available-source run."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    dropped = False
    for entity, columns in outputs.items():
        if entity not in tables:
            continue
        unavailable = [
            column
            for column in columns
            if column in tables[entity] and tables[entity][column].isna().all()
        ]
        if unavailable:
            tables[entity] = tables[entity].drop(columns=unavailable)
            dropped = True
    if not dropped:
        return frame
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _merge_source_operator_outputs(
    pool: Frame,
    operated: Frame,
    outputs: Mapping[str, frozenset[str]],
    *,
    operator_name: str,
) -> tuple[Frame, dict[str, int]]:
    """Merge only one operator's explicit family by entity ID."""

    tables = {entity: pool.table(entity).copy() for entity in pool.entities}
    merged_rows: dict[str, int] = {}
    for entity, columns in outputs.items():
        if not columns:
            continue
        if entity not in pool.entities or entity not in operated.entities:
            raise ValueError(
                f"Multispine source operator {operator_name!r} declares outputs "
                f"for absent entity {entity!r}."
            )
        target = tables[entity]
        source = operated.table(entity)
        entity_id = pool.schema.entity_id_column(entity)
        if entity_id not in target or entity_id not in source:
            raise ValueError(
                f"Multispine source operator {operator_name!r} cannot align "
                f"{entity!r} without {entity_id!r}."
            )
        if source[entity_id].duplicated().any():
            raise ValueError(
                f"Multispine source operator {operator_name!r} returned duplicate "
                f"{entity_id!r} values."
            )
        missing_outputs = sorted(set(columns) - set(source.columns))
        if missing_outputs:
            raise ValueError(
                f"Multispine source operator {operator_name!r} did not emit its "
                f"declared {entity!r} output(s): {missing_outputs}."
            )
        source_by_id = source.set_index(entity_id)
        target_ids = target[entity_id]
        eligible = target_ids.isin(source_by_id.index)
        if int(eligible.sum()) != len(source):
            raise ValueError(
                f"Multispine source operator {operator_name!r} output IDs do not "
                f"align one-to-one with the {entity!r} pool."
            )
        for column in sorted(columns):
            aligned = source_by_id[column].reindex(target_ids)
            if column not in target:
                target[column] = aligned.to_numpy()
            else:
                positions = np.flatnonzero(eligible.to_numpy())
                target.loc[target.index[positions], column] = aligned.iloc[
                    positions
                ].to_numpy()
        merged_rows[entity] = int(eligible.sum())

    merged = Frame(
        tables,
        pool.schema,
        {entity: pool.weights_for(entity) for entity in pool.weighted_entities},
        pool.strata,
        mass_log=pool.mass_log,
        metadata=pool.metadata,
    )
    return merged, merged_rows


def _frame_row_counts(frame: Frame) -> dict[str, int]:
    return {entity: int(len(frame.table(entity))) for entity in frame.entities}


def derive_multispine_pool_inputs(frame: Frame) -> PoolStageOutput:
    """Complete deterministic post-transfer inputs without reading a spine.

    Schedule D capital-gain distributions are derived once per tax unit from
    the transferred parent inputs, then carried by the first person only when
    the unit has no pre-existing values. Existing non-null values are never
    rewritten. The shared QBI reconciliation then restores its documented
    all-or-nothing identities on the imputed PUF-detail surface.
    """

    with_schedule_d, schedule_d_receipt = _complete_schedule_d_input(frame)
    reconciled = with_us_qbi_input_reconciliation(with_schedule_d)
    return PoolStageOutput(
        reconciled,
        {
            "schedule_d_capital_gain_distributions": schedule_d_receipt,
            "qbi_input_reconciliation": {
                "columns": list(US_QBI_OUTPUT_COLUMNS),
                "operation": "shared_all_or_nothing_identity_reconciliation",
            },
        },
    )


def _complete_schedule_d_input(frame: Frame) -> tuple[Frame, dict[str, object]]:
    person = frame.table("person")
    membership_column = frame.schema.membership_column("tax_unit")
    tax_unit_id_column = frame.schema.entity_id_column("tax_unit")
    source_columns = (
        "long_term_capital_gains_before_response",
        "non_sch_d_capital_gains",
    )
    missing_sources = sorted(
        column for column in source_columns if column not in person
    )
    if missing_sources:
        raise ValueError(
            "Schedule D pool derivation requires transferred parent input(s): "
            f"{missing_sources}."
        )
    numeric_sources = person.loc[:, list(source_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    source_values = numeric_sources.to_numpy(dtype=np.float64)
    if not np.isfinite(source_values).all():
        raise ValueError(
            "Schedule D pool derivation requires finite transferred parent inputs."
        )

    tax_unit_ids = frame.table("tax_unit")[tax_unit_id_column]
    grouped = numeric_sources.groupby(
        person[membership_column],
        sort=False,
    ).sum()
    grouped = grouped.reindex(tax_unit_ids.to_numpy())
    if grouped.isna().any().any():
        raise ValueError(
            "Schedule D pool derivation could not align every tax unit to people."
        )
    derived, derivation_receipt = derive_acs_schedule_d_capital_gain_distributions(
        grouped
    )
    derived_by_tax_unit = dict(zip(tax_unit_ids.to_numpy(), derived, strict=True))

    output_column = "schedule_d_capital_gain_distributions"
    if output_column in person:
        output = person[output_column].copy()
        observed = output.notna()
        observed_numeric = pd.to_numeric(output.loc[observed], errors="coerce")
        if not np.isfinite(observed_numeric.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                "Schedule D pool derivation cannot preserve non-finite existing values."
            )
    else:
        output = pd.Series(np.nan, index=person.index, dtype=np.float64)
        observed = pd.Series(False, index=person.index)

    derived_units = 0
    partially_observed_units = 0
    filled_rows = 0
    for tax_unit_id, row_indices in person.groupby(
        membership_column,
        sort=False,
    ).groups.items():
        indices = list(row_indices)
        missing = [index for index in indices if not bool(observed.loc[index])]
        if not missing:
            continue
        if len(missing) == len(indices):
            output.loc[missing[0]] = derived_by_tax_unit[tax_unit_id]
            if len(missing) > 1:
                output.loc[missing[1:]] = 0.0
            derived_units += 1
        else:
            output.loc[missing] = 0.0
            partially_observed_units += 1
        filled_rows += len(missing)

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][output_column] = pd.to_numeric(output, errors="raise").to_numpy(
        dtype=np.float64
    )
    completed = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return completed, {
        "entity": "person",
        "source_grain": "tax_unit",
        "source_columns": list(source_columns),
        "preserved_nonnull_rows": int(observed.sum()),
        "filled_rows": filled_rows,
        "derived_tax_units": derived_units,
        "partially_observed_tax_units_filled_with_zero": partially_observed_units,
        "derivation": derivation_receipt,
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
    prepare_clone: PoolOperator | None = None,
    impute: PoolOperator,
    derive: PoolOperator,
    seed: PoolOperator,
    simulate: PoolOperator,
    agreement_gate: AgreementGate | None = None,
) -> MultispinePoolResult:
    """Run the fixed assembly-to-agreement path over two peer source frames.

    ``prepare_clone`` is the source-blind preparation subphase of ``clone`` and
    receives the assembled all-native pool. ``impute``, ``derive``, and
    ``seed`` each receive the entire physically cloned pool.
    They have no source label argument and are checked at their output boundary
    against the immutable assembly receipt. ``simulate`` returns a temporary
    evaluation frame: formula-owned outputs on that copy are visible to the
    terminal gate but never enter :attr:`MultispinePoolResult.frame`.

    ``agreement_gate`` is an injection seam for small synthetic tests only.
    Production callers omit it, which invokes
    :func:`~populace.build.us_runtime.spine_agreement.spine_agreement_gate`
    with the immutable pool-specific registry and fixed tolerances.
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

    receipts: dict[str, Mapping[str, object]] = {}
    clone_input = assembled
    clone_preparation_receipt: Mapping[str, object] = {}
    if prepare_clone is not None:
        if not callable(prepare_clone):
            raise TypeError("Pool prepare_clone operator must be callable.")
        prepared = prepare_clone(clone_input)
        if not isinstance(prepared, PoolStageOutput):
            raise TypeError(
                "Pool prepare_clone operator must return PoolStageOutput, got "
                f"{type(prepared).__name__}."
            )
        clone_input = prepared.frame
        validate_assembly_provenance(
            clone_input,
            boundary="multispine pool clone preparation output",
        )
        clone_preparation_receipt = dict(prepared.receipt)

    clone_input_rows = _frame_row_counts(clone_input)
    current = clone_us_frame_for_puf_support(clone_input)
    validate_assembly_provenance(
        current,
        boundary="multispine pool clone output",
    )

    if prepare_clone is not None:
        receipts["clone"] = {
            "source_preparation": clone_preparation_receipt,
            "physical_clone": {
                "input_rows": clone_input_rows,
                "output_rows": _frame_row_counts(current),
            },
        }
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

    agreement = (
        spine_agreement_gate(
            simulated.frame,
            registry=POOL_SPINE_AGREEMENT_REGISTRY,
        )
        if agreement_gate is None
        else agreement_gate(simulated.frame)
    )
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
