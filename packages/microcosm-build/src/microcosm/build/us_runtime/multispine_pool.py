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

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import version
from typing import Protocol

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.build.source_runtime import SourceRNGCapability
from microcosm.build.spec_engine.brokers import RNGInvocation
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.us_runtime.acs_income_universe import (
    ACS_PUMS_EARNINGS_UNIVERSE_PERSON_INPUTS,
)
from microcosm.build.us_runtime.acs_transfer import (
    ACS_NATIVE_PERSON_INPUTS,
    TargetFamilies,
    declared_acs_transfer_target_families,
    derive_acs_schedule_d_capital_gain_distributions,
)
from microcosm.build.us_runtime.adult_care import with_us_adult_care_inputs
from microcosm.build.us_runtime.child_support import with_us_child_support_inputs
from microcosm.build.us_runtime.childcare import with_us_childcare_inputs
from microcosm.build.us_runtime.cps_carried import derive_us_cps_carried_inputs
from microcosm.build.us_runtime.disability_benefits import (
    with_us_disability_benefits,
)
from microcosm.build.us_runtime.education_inputs import with_us_education_inputs
from microcosm.build.us_runtime.eligibility_inputs import (
    with_us_eligibility_inputs,
)
from microcosm.build.us_runtime.energy_subsidy import (
    with_us_energy_subsidy_input,
)
from microcosm.build.us_runtime.hours_worked import (
    US_HOURS_WORKED_POOL_EXCLUDED_COLUMNS,
    us_hours_worked_signal_gate,
    with_us_hours_worked_inputs,
)
from microcosm.build.us_runtime.housing_inputs import (
    US_HOUSING_ASSISTANCE_PUF_MAX_TRAIN_SAMPLES,
    US_HOUSING_ASSISTANCE_PUF_N_ESTIMATORS,
    impute_us_housing_assistance_to_puf_support,
    with_us_housing_inputs,
)
from microcosm.build.us_runtime.immigration import with_us_immigration_inputs
from microcosm.build.us_runtime.medicare_take_up import (
    with_us_medicare_take_up_input,
)
from microcosm.build.us_runtime.operator_boundary import (
    FORMULA_OWNED_SOURCE_COLUMNS,
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.pool_physical_authority import (
    RemainingStagePhysicalAuthority,
    SimulationSettings,
    TakeUpPhysicalAuthority,
    TakeUpProgramAuthority,
)
from microcosm.build.us_runtime.pregnancy import with_us_pregnancy_inputs
from microcosm.build.us_runtime.prior_year_income import (
    with_us_prior_year_income_inputs,
)
from microcosm.build.us_runtime.puf_qrf_chain import PRIMARY_QRF_TARGET_ORDER
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.qbi_inputs import (
    US_QBI_RECONCILED_PERSON_COLUMNS,
    bind_us_qbi_reconciliation_transition_authority,
    us_qbi_post_reconciliation_person_columns,
    us_qbi_reconciliation_change_receipt,
    validate_us_qbi_reconciliation_live_output,
    validate_us_qbi_reconciliation_transition,
    with_us_qbi_input_reconciliation,
)
from microcosm.build.us_runtime.relationship_inputs import (
    with_us_relationship_inputs,
)
from microcosm.build.us_runtime.retirement_contributions import (
    with_us_retirement_contribution_inputs,
)
from microcosm.build.us_runtime.retirement_distributions import (
    with_us_retirement_distribution_inputs,
)
from microcosm.build.us_runtime.spine_agreement import (
    default_spine_agreement_registry,
    spine_agreement_gate,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    has_support_role_metadata,
    spine_assembly_receipt,
    spine_provenance_counts,
    support_clone_index_column,
    support_role_series,
    support_source_id_column,
    validate_assembly_provenance,
    without_support_role_metadata,
)
from microcosm.build.us_runtime.take_up import with_us_take_up_inputs
from microcosm.build.us_runtime.take_up_contract import (
    TakeUpProgram,
    load_take_up_contract,
)
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    US_LATE_EDUCATION_NOOP_TARGETS,
    US_LATE_RETIREMENT_SOURCE_MIRROR_TARGETS,
    us_late_overlap_ownership_receipt,
)
from microcosm.build.us_runtime.weeks_unemployed import with_us_weeks_unemployed
from microcosm.build.us_runtime.wic_claim import with_us_wic_claim_input
from microcosm.build.us_runtime.workers_compensation import (
    with_us_workers_compensation,
)
from microcosm.frame import US_SCHEMA, Frame
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
    VariableDependencyClosure,
)

__all__ = [
    "POOL_CHECKPOINT_STAGE_ORDER",
    "POOL_ENGINE_INPUT_PROJECTION_CONTRACT",
    "POOL_HOUSEHOLD_MASS_SHARES",
    "POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES",
    "POOL_HOUSING_ASSISTANCE_N_ESTIMATORS",
    "POOL_DERIVE_OPERATOR_ORDER",
    "POOL_DEFERRED_TRANSFER_INPUTS",
    "POOL_DEFERRED_TRANSFER_STATUS",
    "POOL_OPERATOR_CONTRACTS",
    "POOL_OPERATOR_ORDER",
    "POOL_RANDOM_SEED",
    "POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256",
    "POOL_SSI_DEPENDENCY_CONTRACT",
    "POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE",
    "POOL_POST_CLONE_SOURCE_OPERATOR_ORDER",
    "POOL_POST_CLONE_SOURCE_PHASE",
    "POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER",
    "POOL_SOURCE_OPERATOR_CONTRACTS",
    "POOL_SOURCE_OPERATOR_ORDER",
    "POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE",
    "POOL_SPINE_AGREEMENT_REGISTRY",
    "POOL_TIME_PERIOD",
    "MultispinePoolCheckpoint",
    "MultispinePoolResult",
    "PoolInputSurfaceEntry",
    "PoolEngineInputProjectionContract",
    "PoolRemainingStageInput",
    "PoolSsiDependencyContract",
    "PoolStageOutput",
    "SourceOperatorContract",
    "SourceOperatorRNGDescriptor",
    "complete_multispine_source_inputs",
    "derive_multispine_pool_inputs",
    "finalize_multispine_source_inputs",
    "materialize_multispine_agreement_outputs",
    "materialize_pool_deferred_transfer_inputs",
    "pool_input_surface",
    "pool_engine_input_projection_receipt",
    "pool_remaining_stage_input_manifest",
    "pool_remaining_stage_input_manifest_receipt",
    "pool_ssi_dependency_closure",
    "pool_post_puf_puf_producer_target_families",
    "pool_post_puf_source_producer_target_families",
    "pool_post_puf_transfer_target_families",
    "pool_pre_clone_gap_fill_target_families",
    "pool_transfer_target_families",
    "prepare_multispine_puf_predictors",
    "prepare_multispine_source_inputs_for_clone",
    "run_multispine_post_clone_source_operator",
    "source_operator_rng_invocation_plan",
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

POOL_CHECKPOINT_STAGE_ORDER = ("assembled", "transferred", "simulated")
"""Durable pool states in their fixed resume-precedence order."""

POOL_SOURCE_OPERATOR_ORDER = (
    "derive_us_cps_carried_inputs",
    "with_us_hours_worked_inputs",
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

POOL_DERIVE_OPERATOR_ORDER = (
    "_complete_schedule_d_input",
    "with_us_qbi_input_reconciliation",
)
"""Whole-pool deterministic operators owned by the derive stage."""

POOL_RANDOM_SEED = 0
"""Fixed seed shared by pool imputations and seeded input stages."""

POOL_TIME_PERIOD = 2024
"""PolicyEngine period of the 2024 source pool."""

POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE = False
"""Source construction never reuses an unreceipted pre-existing surface."""

POOL_HOUSING_ASSISTANCE_N_ESTIMATORS = US_HOUSING_ASSISTANCE_PUF_N_ESTIMATORS
POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES = US_HOUSING_ASSISTANCE_PUF_MAX_TRAIN_SAMPLES
"""Exact direct-QRF controls for the housing-assistance source producer."""

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
class PoolInputSurfaceEntry:
    """One normalized member of the pre-simulation pool input surface.

    ``provenance`` names every source registry that requires the same
    variable/entity pair. Overlaps retain the family from the first registry
    in production precedence order: transfer, deferred, primary QRF, then
    take-up contract.
    """

    variable: str
    entity: str
    family: str
    provenance: tuple[str, ...]


@dataclass(frozen=True, order=True)
class PoolRemainingStageInput:
    """One statically declared read after the transferred checkpoint.

    ``provision`` names the producer or fallback doctrine that makes the read
    valid by ``available_by``.  Pseudo-columns enclosed in angle brackets are
    structural Frame resources rather than persisted PolicyEngine variables.
    """

    stage: str
    consumer: str
    entity: str
    variable: str
    execution_scope: str
    provision: str
    available_by: str
    fallback: str | None = None


@dataclass(frozen=True)
class PoolSsiDependencyContract:
    """Checked-in identity of the static PE-US SSI dependency closure."""

    engine_version: str
    root: str
    input_leaf_count: int
    formula_node_count: int
    edge_count: int
    sha256: str


@dataclass(frozen=True)
class PoolEngineInputProjectionContract:
    """Pinned identity of every installed engine input scanned at simulate."""

    engine_version: str
    input_count: int
    default_count: int
    sha256: str
    defaults_sha256: str


POOL_SSI_DEPENDENCY_CONTRACT = PoolSsiDependencyContract(
    engine_version="1.764.6",
    root="ssi",
    input_leaf_count=55,
    formula_node_count=62,
    edge_count=186,
    sha256="e3351cdedbe592456b637286ecd04b7079746e1c409e594fbca60a7d28666838",
)
"""Exact static graph consumed by the terminal SSI agreement simulation."""

POOL_ENGINE_INPUT_PROJECTION_CONTRACT = PoolEngineInputProjectionContract(
    engine_version="1.764.6",
    input_count=863,
    default_count=863,
    sha256="67a66b018c6261a03a88852cce5c5a4cbe9f5595735d17f2f7666e19e464dfbf",
    defaults_sha256="87f508fbb382036946aa5e225d339e1b593a464ee6cfc644d7d710540b00a9a7",
)
"""Exact installed input registry scanned by the disposable projection."""

POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256 = (
    "8247a93e5f8f63d3ae71c1de681c29524d4bb8f07e3c6a50dcaf431b1377020f"
)
"""Pinned content digest of all 993 post-transfer consumer/input rows."""


@dataclass(frozen=True)
class PoolStageOutput:
    """One source-blind operator result and its manifest-ready receipt."""

    frame: Frame
    receipt: Mapping[str, object] = field(default_factory=dict)
    qbi_transition_authority_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError(
                "PoolStageOutput.frame must be a Frame, got "
                f"{type(self.frame).__name__}."
            )
        if not isinstance(self.receipt, Mapping):
            raise TypeError("PoolStageOutput.receipt must be a mapping.")
        if self.qbi_transition_authority_sha256 is not None and not isinstance(
            self.qbi_transition_authority_sha256,
            str,
        ):
            raise TypeError(
                "PoolStageOutput.qbi_transition_authority_sha256 must be a "
                "string when present."
            )


@dataclass(frozen=True)
class MultispinePoolCheckpoint:
    """One validated in-memory pool state suitable for durable serialization.

    ``frame`` is always the input-only state that production may eventually
    publish. Only the terminal ``simulated`` checkpoint also carries the
    disposable formula-output view used by the always-fresh agreement gate.
    """

    stage: str
    frame: Frame
    assembly_receipt: Mapping[str, object]
    stage_receipts: Mapping[str, Mapping[str, object]]
    simulation_frame: Frame | None = None
    qbi_transition_authority_sha256: str | None = None
    late_producer_transition_authority_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in POOL_CHECKPOINT_STAGE_ORDER:
            raise ValueError(
                f"Unknown multispine pool checkpoint stage {self.stage!r}; "
                f"expected one of {POOL_CHECKPOINT_STAGE_ORDER}."
            )
        if not isinstance(self.frame, Frame):
            raise TypeError(
                "MultispinePoolCheckpoint.frame must be a Frame, got "
                f"{type(self.frame).__name__}."
            )
        if not isinstance(self.assembly_receipt, Mapping):
            raise TypeError(
                "MultispinePoolCheckpoint.assembly_receipt must be a mapping."
            )
        if not isinstance(self.stage_receipts, Mapping) or any(
            not isinstance(name, str) or not isinstance(receipt, Mapping)
            for name, receipt in self.stage_receipts.items()
        ):
            raise TypeError(
                "MultispinePoolCheckpoint.stage_receipts must map stage names "
                "to receipt mappings."
            )
        if self.stage == "simulated":
            if not isinstance(self.simulation_frame, Frame):
                raise TypeError(
                    "A simulated multispine pool checkpoint requires a "
                    "simulation_frame."
                )
        elif self.simulation_frame is not None:
            raise ValueError(
                "Only a simulated multispine pool checkpoint may carry a "
                "simulation_frame."
            )
        if self.qbi_transition_authority_sha256 is not None and not isinstance(
            self.qbi_transition_authority_sha256,
            str,
        ):
            raise TypeError(
                "MultispinePoolCheckpoint.qbi_transition_authority_sha256 must "
                "be a string when present."
            )
        if (
            self.late_producer_transition_authority_sha256 is not None
            and not isinstance(
                self.late_producer_transition_authority_sha256,
                str,
            )
        ):
            raise TypeError(
                "MultispinePoolCheckpoint."
                "late_producer_transition_authority_sha256 must be a string "
                "when present."
            )


@dataclass(frozen=True)
class MultispinePoolResult:
    """Input-only pool plus receipts from its terminal agreement evaluation."""

    frame: Frame
    assembly_receipt: Mapping[str, object]
    provenance_counts: Mapping[str, Mapping[str, object]]
    stage_receipts: Mapping[str, Mapping[str, object]]
    agreement_gate: GateResult
    qbi_transition_authority_sha256: str | None = None

    @property
    def simulation_ready(self) -> bool:
        """Whether the unchanged terminal agreement gate passed."""

        return self.agreement_gate.passed


type PoolOperator = Callable[[Frame], PoolStageOutput]
type AgreementGate = Callable[[Frame], GateResult]
type SourceFrameOperator = Callable[[Frame], Frame | PoolStageOutput]


@dataclass(frozen=True)
class SourceOperatorContract:
    """Clone-phase placement and the mechanism that requires it."""

    family: str
    phases: tuple[str, ...]
    mechanism: str
    execution_scope: str = "cps_source"


@dataclass(frozen=True)
class SourceOperatorRNGDescriptor:
    """Exact ledger sites a post-clone source callback may consume."""

    site_ids: tuple[str, ...]
    training_cap_site: str | None = None
    training_cap: int | None = None
    qrf_site: str | None = None
    stable_draw_sites: tuple[str, ...] = ()
    generator_site: str | None = None
    pre_clone_overgrants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.site_ids or len(self.site_ids) != len(set(self.site_ids)):
            raise ValueError("Source RNG descriptor sites must be nonempty and unique.")
        referenced = {
            *self.stable_draw_sites,
            *self.pre_clone_overgrants,
            *(
                (self.training_cap_site,)
                if self.training_cap_site is not None
                else ()
            ),
            *((self.qrf_site,) if self.qrf_site is not None else ()),
            *((self.generator_site,) if self.generator_site is not None else ()),
        }
        if referenced != set(self.site_ids):
            raise ValueError(
                "Source RNG descriptor classifications must exactly cover its sites."
            )
        if (self.training_cap_site is None) != (self.training_cap is None):
            raise ValueError(
                "Source RNG training-cap site and positive cap must be declared together."
            )
        if self.training_cap is not None and self.training_cap < 1:
            raise ValueError("Source RNG training cap must be positive.")


_PRE_CLONE_PHASE = "pre_clone"
_POST_CLONE_PHASE = "post_clone"
POOL_POST_CLONE_SOURCE_PHASE = _POST_CLONE_PHASE
_CPS_SOURCE_EXECUTION_SCOPE = "cps_source"
_WHOLE_POOL_EXECUTION_SCOPE = "whole_pool"

POOL_OPERATOR_CONTRACTS: Mapping[str, SourceOperatorContract] = {
    "derive_us_cps_carried_inputs": SourceOperatorContract(
        "cps_carried",
        (_PRE_CLONE_PHASE,),
        "rowwise measured mappings needed by the pre-clone rent predictors",
    ),
    "with_us_hours_worked_inputs": SourceOperatorContract(
        "hours_worked",
        (_PRE_CLONE_PHASE,),
        "direct measured mappings should be cloned once per source person",
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
    "_complete_schedule_d_input": SourceOperatorContract(
        "capital_gain_distributions",
        (_POST_CLONE_PHASE,),
        "transferred tax-unit parents exist only on the physically cloned pool",
        execution_scope=_WHOLE_POOL_EXECUTION_SCOPE,
    ),
    "with_us_qbi_input_reconciliation": SourceOperatorContract(
        "qbi_reconciliation",
        (_POST_CLONE_PHASE,),
        "all-or-nothing identities reconcile the post-transfer PUF detail surface",
        execution_scope=_WHOLE_POOL_EXECUTION_SCOPE,
    ),
}
"""Total clone-phase registry for all 23 pool-path operator kernels."""

POOL_SOURCE_OPERATOR_CONTRACTS = POOL_OPERATOR_CONTRACTS
"""Backward-compatible name for the now-total pool operator registry."""


_POST_CLONE_SOURCE_RNG_DESCRIPTORS: Mapping[
    str, SourceOperatorRNGDescriptor
] = {
    "with_us_prior_year_income_inputs": SourceOperatorRNGDescriptor(
        (
            "prior_year_income_puf_qrf_model",
            "prior_year_income_training_cap",
        ),
        training_cap_site="prior_year_income_training_cap",
        training_cap=5_000,
        qrf_site="prior_year_income_puf_qrf_model",
    ),
    "with_us_pregnancy_inputs": SourceOperatorRNGDescriptor(
        ("pregnancy_assignment",),
        stable_draw_sites=("pregnancy_assignment",),
    ),
    "with_us_wic_claim_input": SourceOperatorRNGDescriptor(
        ("wic_claim_assignment",),
        stable_draw_sites=("wic_claim_assignment",),
    ),
    "impute_us_housing_assistance_to_puf_support": SourceOperatorRNGDescriptor(
        (
            "acs_rent_archived_training_cap",
            "acs_rent_qrf_model",
            "housing_assistance_puf_qrf_model",
            "housing_inputs_training_cap",
        ),
        training_cap_site="housing_inputs_training_cap",
        training_cap=POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES,
        qrf_site="housing_assistance_puf_qrf_model",
        pre_clone_overgrants=(
            "acs_rent_archived_training_cap",
            "acs_rent_qrf_model",
        ),
    ),
    "with_us_child_support_inputs": SourceOperatorRNGDescriptor(
        ("child_support_puf_qrf_model", "child_support_training_cap"),
        training_cap_site="child_support_training_cap",
        training_cap=5_000,
        qrf_site="child_support_puf_qrf_model",
    ),
    "with_us_disability_benefits": SourceOperatorRNGDescriptor(
        (
            "disability_benefits_puf_qrf_model",
            "disability_benefits_training_cap",
        ),
        training_cap_site="disability_benefits_training_cap",
        training_cap=5_000,
        qrf_site="disability_benefits_puf_qrf_model",
    ),
    "with_us_workers_compensation": SourceOperatorRNGDescriptor(
        (
            "workers_compensation_puf_qrf_model",
            "workers_compensation_training_cap",
        ),
        training_cap_site="workers_compensation_training_cap",
        training_cap=5_000,
        qrf_site="workers_compensation_puf_qrf_model",
    ),
    "with_us_weeks_unemployed": SourceOperatorRNGDescriptor(
        ("weeks_unemployed_puf_qrf_model", "weeks_unemployed_training_cap"),
        training_cap_site="weeks_unemployed_training_cap",
        training_cap=5_000,
        qrf_site="weeks_unemployed_puf_qrf_model",
    ),
    "with_us_childcare_inputs": SourceOperatorRNGDescriptor(
        ("childcare_puf_qrf_model", "childcare_training_cap"),
        training_cap_site="childcare_training_cap",
        training_cap=5_000,
        qrf_site="childcare_puf_qrf_model",
    ),
    "with_us_adult_care_inputs": SourceOperatorRNGDescriptor(
        ("adult_care_weighted_prefix_assignment",),
        generator_site="adult_care_weighted_prefix_assignment",
    ),
    "with_us_energy_subsidy_input": SourceOperatorRNGDescriptor(
        ("energy_subsidy_puf_qrf_model", "energy_subsidy_training_cap"),
        training_cap_site="energy_subsidy_training_cap",
        training_cap=5_000,
        qrf_site="energy_subsidy_puf_qrf_model",
    ),
    "with_us_retirement_contribution_inputs": SourceOperatorRNGDescriptor(
        (
            "retirement_contributions_puf_qrf_model",
            "retirement_contributions_training_cap",
        ),
        training_cap_site="retirement_contributions_training_cap",
        training_cap=5_000,
        qrf_site="retirement_contributions_puf_qrf_model",
    ),
    "with_us_retirement_distribution_inputs": SourceOperatorRNGDescriptor(
        (
            "retirement_distributions_puf_qrf_model",
            "retirement_distributions_training_cap",
        ),
        training_cap_site="retirement_distributions_training_cap",
        training_cap=5_000,
        qrf_site="retirement_distributions_puf_qrf_model",
    ),
    "with_us_immigration_inputs": SourceOperatorRNGDescriptor(
        (
            "immigration_ead_workers_assignment",
            "immigration_ead_students_assignment",
        ),
        stable_draw_sites=(
            "immigration_ead_workers_assignment",
            "immigration_ead_students_assignment",
        ),
    ),
}

POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER = tuple(
    name
    for name in POOL_SOURCE_OPERATOR_ORDER
    if _PRE_CLONE_PHASE in POOL_OPERATOR_CONTRACTS[name].phases
)
"""Source operations owned by the clone stage before physical expansion."""

POOL_POST_CLONE_SOURCE_OPERATOR_ORDER = tuple(
    name
    for name in POOL_SOURCE_OPERATOR_ORDER
    if _POST_CLONE_PHASE in POOL_OPERATOR_CONTRACTS[name].phases
)
"""Source operations safe or required after physical support expansion."""


def _source_stable_keys(frame: Frame, operator_name: str) -> list[str]:
    person = frame.table("person")
    if operator_name == "with_us_pregnancy_inputs":
        if {"source_year", "source_household_id", "source_person_id"} <= set(
            person.columns
        ):
            keys = (
                person["source_year"].astype(str)
                + ":"
                + person["source_household_id"].astype(str)
                + ":"
                + person["source_person_id"].astype(str)
            )
        else:
            keys = person["person_id"].astype(str)
    elif operator_name == "with_us_wic_claim_input":
        source_columns = ("source_year", "source_household_id", "source_person_id")
        if all(column in person for column in source_columns):
            identity = person.loc[:, list(source_columns)]
            if identity.isna().any().any():
                raise ValueError("WIC source RNG plan found incomplete stable identity.")
            keys = (
                identity["source_year"].astype(str)
                + ":"
                + identity["source_household_id"].astype(str)
                + ":"
                + identity["source_person_id"].astype(str)
            )
        elif "person_support_source_id" in person:
            source_id = person["person_support_source_id"]
            if source_id.isna().any():
                raise ValueError("WIC source RNG plan found missing support identity.")
            keys = "support:" + source_id.astype(str)
        else:
            keys = "person:" + person["person_id"].astype(str)
    elif operator_name == "with_us_immigration_inputs":
        if {"source_year", "source_person_id"}.issubset(person.columns):
            keys = (
                person["source_year"].astype(str)
                + ":"
                + person["source_person_id"].astype(str)
            )
        else:
            keys = person["person_id"].astype(str)
    else:  # pragma: no cover - closed by the descriptor registry
        raise ValueError(f"{operator_name!r} has no stable-key RNG descriptor.")
    return keys.tolist()


def source_operator_rng_invocation_plan(
    frame: Frame,
    operator_name: str,
    *,
    granted_site_ids: Sequence[str],
) -> Mapping[str, tuple[RNGInvocation, ...]]:
    """Build the exact per-site plan for one post-clone source dispatch.

    The compiled node remains authoritative: its granted site ids must match
    the checked source descriptor exactly.  Every grant receives an explicit
    plan, including empty plans for conditional training caps and the two
    pre-clone ACS-rent sites overgranted to the housing callback's shared
    source-stage owner.
    """

    if not isinstance(frame, Frame):
        raise TypeError("Source RNG invocation planning requires a Frame.")
    try:
        descriptor = _POST_CLONE_SOURCE_RNG_DESCRIPTORS[operator_name]
    except KeyError as error:
        if operator_name in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER:
            if tuple(granted_site_ids):
                raise ValueError(
                    f"Deterministic source operator {operator_name!r} has RNG grants."
                ) from error
            return {}
        raise ValueError(
            f"Unknown post-clone source operator {operator_name!r}."
        ) from error

    granted = tuple(granted_site_ids)
    if len(granted) != len(set(granted)) or set(granted) != set(descriptor.site_ids):
        raise ValueError(
            f"Source RNG grants for {operator_name!r} differ from its exact "
            f"descriptor: expected={list(descriptor.site_ids)!r}, "
            f"observed={list(granted)!r}."
        )

    inactive = False
    person = frame.table("person")
    if operator_name == "with_us_pregnancy_inputs":
        inactive = (
            "is_pregnant" in person
            and person["is_pregnant"].dropna().nunique() > 1
        )
    elif operator_name == "with_us_immigration_inputs":
        inactive = {
            "ssn_card_type",
            "immigration_status_str",
        }.issubset(person.columns)

    stable_material: Mapping[str, object] | None = None
    if descriptor.stable_draw_sites and not inactive:
        stable_material = {
            "stable_keys_sha256": sha256_json(
                [str(key) for key in _source_stable_keys(frame, operator_name)]
            )
        }

    asec_rows = 0
    if descriptor.training_cap_site is not None and not inactive:
        if has_support_role_metadata(person, entity="person"):
            asec_rows = int(
                support_role_series(person, entity="person")
                .eq(BASE_ASEC_SUPPORT_CHANNEL)
                .sum()
            )

    plans: dict[str, tuple[RNGInvocation, ...]] = {}
    for site_id in granted:
        if inactive or site_id in descriptor.pre_clone_overgrants:
            plans[site_id] = ()
        elif site_id == descriptor.training_cap_site:
            assert descriptor.training_cap is not None
            plans[site_id] = (
                RNGInvocation(
                    "default",
                    {"stage_training_cap": descriptor.training_cap},
                ),
            ) if asec_rows > descriptor.training_cap else ()
        elif site_id in descriptor.stable_draw_sites:
            assert stable_material is not None
            plans[site_id] = (RNGInvocation("default", stable_material),)
        elif site_id in {descriptor.qrf_site, descriptor.generator_site}:
            plans[site_id] = (RNGInvocation("default"),)
        else:  # pragma: no cover - descriptor construction is exhaustive
            raise AssertionError(f"Unclassified source RNG site {site_id!r}.")
    return plans

_CPS_SOURCE_EVIDENCE_COLUMN = "PERIDNUM"
_SOURCE_OPERATOR_FAMILIES: Mapping[str, str] = {
    name: contract.family for name, contract in POOL_OPERATOR_CONTRACTS.items()
}
_FORMULA_OWNED_SOURCE_OUTPUTS = FORMULA_OWNED_SOURCE_COLUMNS
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
_POOL_SIMULATION_PRESERVED_ENGINE_INPUTS: Mapping[
    tuple[str, str], tuple[str, str | None]
] = {
    ("person", "is_related_to_head_or_spouse"): ("assembled", None),
    ("household", "puma"): (
        "assembled",
        "ephemeral_simulation_projection_engine_default_for_null",
    ),
    ("person", "ssi_reported"): (
        "transferred",
        "ephemeral_simulation_projection_engine_default_for_null",
    ),
    ("household", "state_fips"): ("assembled", None),
}

_SCF_WEALTH_DEFERRAL_REASON = (
    "The increment-2 pool input contract contains no SCF 2022 or SIPP 2023 "
    "financial-asset donor. The downstream fiscal-refresh SCF-wealth stage "
    "owns this required release input; the pool preserves an explicit null "
    "column until that donor-backed operator runs."
)
POOL_DEFERRED_TRANSFER_INPUTS: Mapping[str, Mapping[str, str]] = {
    column: {
        "entity": "person",
        "legacy_family": "model_required_numeric",
        "owner": "with_us_scf_wealth_inputs",
        "physical_dtype": "float64",
        "reason": _SCF_WEALTH_DEFERRAL_REASON,
    }
    for column in (
        "bank_account_assets",
        "bond_assets",
        "stock_assets",
    )
}
POOL_DEFERRED_TRANSFER_STATUS = "deferred_pending_source_donor"
"""Pool-stage-only deferrals for source inputs whose donors are out of scope.

These remain hard release requirements and remain in the legacy ACS transfer
declaration. The raw-only increment-2 pool has only its six pinned inputs, so
it cannot honestly manufacture the SCF/SIPP asset surface. It instead carries
typed all-null columns and a receipt; engine defaults may be used only on the
disposable SSI agreement view, whose default fills are separately receipted.
"""


def pool_transfer_target_families() -> TargetFamilies:
    """Return the fixed pool-only raw-preserving QRF transfer plan.

    The legacy declaration remains unchanged. This pool-local copy excludes
    only the explicitly receipted SCF/SIPP asset deferrals, then adds every
    persisted historical source-operator output that ACS does not map natively
    and the legacy plan does not already own. A target appears in exactly one
    family, so transfer provenance stays unambiguous. Formula-owned outputs are
    never transfer targets.

    The #581 agreement registry supplements this plan with the complete take-up
    inventory and formula-owned SSI. Take-up inputs not owned by QRF remain in
    the later seed stage, where sourced draws and disclosed engine defaults
    remain distinguishable in the receipt.
    """

    deferred = frozenset(POOL_DEFERRED_TRANSFER_INPUTS)
    plan: dict[str, dict[str, tuple[str, ...]]] = {
        entity: {
            family: tuple(column for column in columns if column not in deferred)
            for family, columns in families.items()
        }
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


def _partition_pool_transfer_target_families(
    target_families: TargetFamilies,
) -> tuple[TargetFamilies, TargetFamilies]:
    """Split the legacy transfer surface at its declared producer boundary.

    Targets emitted by a pre-clone source operator are eligible for the early
    cross-origin gap-fill. Every other target is produced only by the primary
    PUF pass or post-clone source completion and therefore belongs to the
    post-PUF transfer. The partition preserves the legacy entity, family, and
    target order so fit identities stay deterministic.
    """

    pre_clone_outputs = {
        (entity, column)
        for operator_name in POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            POOL_OPERATOR_CONTRACTS[operator_name].family
        ].items()
        for column in columns
    }
    early: dict[str, dict[str, tuple[str, ...]]] = {}
    late: dict[str, dict[str, tuple[str, ...]]] = {}
    for entity, families in target_families.items():
        for family, targets in families.items():
            early_targets = tuple(
                target for target in targets if (entity, target) in pre_clone_outputs
            )
            late_targets = tuple(
                target
                for target in targets
                if (entity, target) not in pre_clone_outputs
            )
            if early_targets:
                early.setdefault(entity, {})[family] = early_targets
            if late_targets:
                late.setdefault(entity, {})[family] = late_targets
    return early, late


def pool_pre_clone_gap_fill_target_families() -> TargetFamilies:
    """Return targets available after native pre-clone source preparation."""

    early, _late = _partition_pool_transfer_target_families(
        pool_transfer_target_families()
    )
    return early


def pool_post_puf_transfer_target_families() -> TargetFamilies:
    """Return targets first available after PUF and source completion."""

    _early, late = _partition_pool_transfer_target_families(
        pool_transfer_target_families()
    )
    return late


def _filter_target_families_by_outputs(
    target_families: TargetFamilies,
    outputs: set[tuple[str, str]],
) -> TargetFamilies:
    """Preserve declaration order while retaining targets with named producers."""

    filtered: dict[str, dict[str, tuple[str, ...]]] = {}
    for entity, families in target_families.items():
        for family, targets in families.items():
            produced = tuple(
                target for target in targets if (entity, target) in outputs
            )
            if produced:
                filtered.setdefault(entity, {})[family] = produced
    return filtered


def pool_post_puf_puf_producer_target_families() -> TargetFamilies:
    """Return late targets whose producer role is a live PUF clone."""

    puf_outputs = {
        (entity, target)
        for entity, targets in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            "primary_puf_qrf"
        ].items()
        for target in targets
    }
    return _filter_target_families_by_outputs(
        pool_post_puf_transfer_target_families(),
        puf_outputs,
    )


def pool_post_puf_source_producer_target_families() -> TargetFamilies:
    """Return late targets whose producer role is an ASEC source clone."""

    source_outputs = {
        (entity, target)
        for operator_name in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        for entity, targets in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            POOL_OPERATOR_CONTRACTS[operator_name].family
        ].items()
        for target in targets
    }
    return _filter_target_families_by_outputs(
        pool_post_puf_transfer_target_families(),
        source_outputs,
    )


def _resolve_take_up_program_bindings(
    bindings: tuple[tuple[str, str, str], ...] | None,
) -> tuple[tuple[str, str, str], ...]:
    """Return ``(variable, entity, treatment)`` rows for static manifests."""

    if bindings is None:
        from microcosm.build.spec_engine.engine_abi import (
            active_take_up_manifest_program_bindings,
        )

        bindings = active_take_up_manifest_program_bindings()
    if bindings is None:
        return tuple(
            (program.variable, program.entity, program.populace_treatment)
            for program in load_take_up_contract().programs
        )
    for index, binding in enumerate(bindings):
        if (
            len(binding) != 3
            or not all(isinstance(value, str) and value for value in binding)
        ):
            raise ValueError(
                "Take-up manifest program binding must contain three non-empty "
                f"strings at index {index}."
            )
    return bindings


def pool_input_surface(
    *,
    take_up_program_bindings: tuple[tuple[str, str, str], ...] | None = None,
) -> tuple[PoolInputSurfaceEntry, ...]:
    """Return the complete registry-derived pool input/imputation surface.

    The surface is deliberately limited to transfer targets, explicit deferred
    inputs, the production primary-QRF target order, and every checked-in
    take-up contract variable. Variables repeated across registries are merged
    by variable/entity and carry every provenance receipt. A variable assigned
    to conflicting entities fails closed.
    """

    entries: dict[tuple[str, str], PoolInputSurfaceEntry] = {}
    entity_by_variable: dict[str, str] = {}

    def register(
        variable: str,
        *,
        entity: str,
        family: str,
        provenance: str,
    ) -> None:
        previous_entity = entity_by_variable.setdefault(variable, entity)
        if previous_entity != entity:
            raise ValueError(
                f"Pool input {variable!r} has conflicting entities "
                f"{previous_entity!r} and {entity!r}."
            )
        key = (variable, entity)
        existing = entries.get(key)
        if existing is None:
            entries[key] = PoolInputSurfaceEntry(
                variable=variable,
                entity=entity,
                family=family,
                provenance=(provenance,),
            )
            return
        if provenance not in existing.provenance:
            entries[key] = PoolInputSurfaceEntry(
                variable=existing.variable,
                entity=existing.entity,
                family=existing.family,
                provenance=(*existing.provenance, provenance),
            )

    for entity, families in pool_transfer_target_families().items():
        for family, variables in families.items():
            for variable in variables:
                register(
                    variable,
                    entity=entity,
                    family=family,
                    provenance="pool_transfer_target_families",
                )

    for variable, declaration in POOL_DEFERRED_TRANSFER_INPUTS.items():
        register(
            variable,
            entity=declaration["entity"],
            family="deferred_asset",
            provenance="POOL_DEFERRED_TRANSFER_INPUTS",
        )

    primary_qrf_outputs = PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["primary_puf_qrf"]
    for variable in PRIMARY_QRF_TARGET_ORDER:
        entities = sorted(
            entity
            for entity, variables in primary_qrf_outputs.items()
            if variable in variables
        )
        if not entities:
            raise ValueError(
                f"Primary QRF pool input {variable!r} has no declared entity."
            )
        if len(entities) != 1:
            raise ValueError(
                f"Primary QRF pool input {variable!r} has conflicting entities "
                f"{entities}."
            )
        register(
            variable,
            entity=entities[0],
            family="primary_puf_qrf_nontransfer",
            provenance="PRIMARY_QRF_TARGET_ORDER",
        )

    for variable, entity, populace_treatment in _resolve_take_up_program_bindings(
        take_up_program_bindings
    ):
        register(
            variable,
            entity=entity,
            family=f"take_up_{populace_treatment}",
            provenance="load_take_up_contract",
        )

    return tuple(
        sorted(entries.values(), key=lambda entry: (entry.variable, entry.entity))
    )


def pool_ssi_dependency_closure(
    metadata_index: PolicyEngineUSVariableMetadataIndex | None = None,
) -> VariableDependencyClosure:
    """Return SSI's static PE-US graph after checking the pinned identity."""

    index = (
        metadata_index
        if metadata_index is not None
        else PolicyEngineUSVariableMetadataIndex()
    )
    closure = index.variable_dependency_closure(POOL_SSI_DEPENDENCY_CONTRACT.root)
    observed = {
        "engine_version": closure.engine_version,
        "root": closure.root,
        "input_leaf_count": len(closure.input_leaves),
        "formula_node_count": len(closure.formula_nodes),
        "edge_count": len(closure.edges),
        "sha256": closure.sha256,
    }
    expected = {
        "engine_version": POOL_SSI_DEPENDENCY_CONTRACT.engine_version,
        "root": POOL_SSI_DEPENDENCY_CONTRACT.root,
        "input_leaf_count": POOL_SSI_DEPENDENCY_CONTRACT.input_leaf_count,
        "formula_node_count": POOL_SSI_DEPENDENCY_CONTRACT.formula_node_count,
        "edge_count": POOL_SSI_DEPENDENCY_CONTRACT.edge_count,
        "sha256": POOL_SSI_DEPENDENCY_CONTRACT.sha256,
    }
    if observed != expected:
        raise ValueError(
            "PolicyEngine-US SSI dependency closure drifted; refresh the "
            f"remaining-stage input audit. expected={expected}, observed={observed}."
        )
    return closure


def _pool_engine_input_projection(
    metadata_index: PolicyEngineUSVariableMetadataIndex,
    *,
    engine_version: str,
) -> tuple[tuple[str, str], ...]:
    """Return every installed engine input after checking its pinned digest."""

    projection = tuple(
        (metadata_index.variable_metadata(variable).entity, variable)
        for variable in metadata_index.variables()
    )
    payload = [
        {"entity": entity, "variable": variable} for entity, variable in projection
    ]
    digest = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    observed = {
        "engine_version": engine_version,
        "input_count": len(projection),
        "sha256": digest,
    }
    expected = {
        "engine_version": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.engine_version,
        "input_count": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.input_count,
        "sha256": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.sha256,
    }
    if observed != expected:
        raise ValueError(
            "PolicyEngine-US simulation input projection drifted; refresh the "
            f"remaining-stage input audit. expected={expected}, observed={observed}."
        )
    return projection


def pool_engine_input_projection_receipt(
    engine: _PoolRulesEngine | None = None,
) -> dict[str, object]:
    """Validate every installed simulation input and its declared default."""

    rules_engine = engine
    if rules_engine is None:
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        rules_engine = PolicyEngineUSEngine()
    variables = list(rules_engine.variables())
    defaults = dict(rules_engine.default_values(variables))
    missing_defaults = sorted(set(variables) - set(defaults))
    extra_defaults = sorted(set(defaults) - set(variables))
    if missing_defaults or extra_defaults:
        raise ValueError(
            "PolicyEngine-US simulation input default surface is not exact; "
            f"missing={missing_defaults}, extra={extra_defaults}."
        )
    rows = [
        {
            "entity": rules_engine.variable_metadata(variable).entity,
            "variable": variable,
            "default": defaults[variable],
        }
        for variable in variables
    ]
    defaults_sha256 = hashlib.sha256(
        json.dumps(
            rows,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    observed = {
        "engine_version": version("policyengine-us"),
        "input_count": len(variables),
        "default_count": len(defaults),
        "defaults_sha256": defaults_sha256,
    }
    expected = {
        "engine_version": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.engine_version,
        "input_count": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.input_count,
        "default_count": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.default_count,
        "defaults_sha256": (POOL_ENGINE_INPUT_PROJECTION_CONTRACT.defaults_sha256),
    }
    if observed != expected:
        raise ValueError(
            "PolicyEngine-US simulation input defaults drifted; refresh the "
            f"remaining-stage input audit. expected={expected}, observed={observed}."
        )
    return observed


@lru_cache(maxsize=8)
def pool_remaining_stage_input_manifest(
    metadata_index: PolicyEngineUSVariableMetadataIndex | None = None,
    *,
    take_up_program_bindings: tuple[tuple[str, str, str], ...] | None = None,
) -> tuple[PoolRemainingStageInput, ...]:
    """Enumerate and statically provision every remaining-stage data read.

    The manifest starts at a validated ``transferred`` checkpoint and covers
    tail preparation, both derive kernels, all thirteen seed-program branches,
    and the terminal SSI simulation.  Simulation leaves come from the pinned
    source-index graph rather than a handwritten dependency list.
    """

    entries: dict[
        tuple[str, str, str, str],
        PoolRemainingStageInput,
    ] = {}

    def register(
        stage: str,
        consumer: str,
        entity: str,
        variable: str,
        *,
        execution_scope: str,
        provision: str,
        available_by: str,
        fallback: str | None = None,
    ) -> None:
        entry = PoolRemainingStageInput(
            stage=stage,
            consumer=consumer,
            entity=entity,
            variable=variable,
            execution_scope=execution_scope,
            provision=provision,
            available_by=available_by,
            fallback=fallback,
        )
        key = (stage, consumer, entity, variable)
        previous = entries.setdefault(key, entry)
        if previous != entry:
            raise ValueError(
                "Remaining-stage input has conflicting provisions: "
                f"{previous!r} versus {entry!r}."
            )

    program_bindings = _resolve_take_up_program_bindings(take_up_program_bindings)
    surface = {
        entry.variable: entry
        for entry in pool_input_surface(take_up_program_bindings=program_bindings)
    }

    def surface_provision(variable: str) -> str:
        declaration = surface.get(variable)
        if declaration is None:
            raise ValueError(
                f"Remaining-stage input {variable!r} has no pool input producer."
            )
        return f"pool_input_surface:{declaration.family}"

    # The stacked tail step reads only clone provenance and the optional memo
    # leaf that it deliberately clears before deterministic re-derivation.
    register(
        "derive",
        "prepare_stacked_tail_derivation",
        "person",
        support_clone_index_column("person"),
        execution_scope="whole_pool",
        provision="assembly_support_provenance",
        available_by="assembled",
    )
    register(
        "derive",
        "prepare_stacked_tail_derivation",
        "person",
        "schedule_d_capital_gain_distributions",
        execution_scope="clone_2",
        provision="optional_existing_derived_leaf",
        available_by="transferred",
        fallback="absent_or_cleared_then_schedule_d_derived",
    )

    for variable in (
        "long_term_capital_gains_before_response",
        "non_sch_d_capital_gains",
    ):
        register(
            "derive",
            "_complete_schedule_d_input",
            "person",
            variable,
            execution_scope="whole_pool",
            provision=surface_provision(variable),
            available_by="transferred",
        )
    for entity, variable, provision in (
        ("person", "person_tax_unit_id", "frame_membership"),
        ("tax_unit", "tax_unit_id", "frame_entity_id"),
    ):
        register(
            "derive",
            "_complete_schedule_d_input",
            entity,
            variable,
            execution_scope="whole_pool",
            provision=provision,
            available_by="assembled",
        )
    register(
        "derive",
        "_complete_schedule_d_input",
        "person",
        "schedule_d_capital_gain_distributions",
        execution_scope="whole_pool",
        provision="optional_transferred_or_schedule_d_derived",
        available_by="transferred",
        fallback="derive_from_finite_transferred_parents",
    )

    for variable in US_QBI_RECONCILED_PERSON_COLUMNS:
        register(
            "derive",
            "with_us_qbi_input_reconciliation",
            "person",
            variable,
            execution_scope="whole_pool",
            provision=surface_provision(variable),
            available_by="transferred",
        )
    for variable in (
        "partnership_income",
        "estate_income",
        "non_qualified_dividend_income",
    ):
        register(
            "derive",
            "with_us_qbi_input_reconciliation",
            "person",
            variable,
            execution_scope="whole_pool",
            provision=surface_provision(variable),
            available_by="transferred",
        )
    register(
        "derive",
        "with_us_qbi_input_reconciliation",
        "person",
        "s_corp_income",
        execution_scope="whole_pool",
        provision="primary_puf_exact_zero_universe",
        available_by="transferred",
    )
    for variable, provision in (
        ("age", "assembled_native_person_input"),
        ("SEMP", "assembled_raw_acs_source_authority"),
        ("person_tax_unit_id", "frame_membership"),
        (support_clone_index_column("person"), "assembly_support_provenance"),
    ):
        register(
            "derive",
            "with_us_qbi_input_reconciliation",
            "person",
            variable,
            execution_scope="whole_pool",
            provision=provision,
            available_by="assembled",
        )
    universe_owner_inputs = ACS_PUMS_EARNINGS_UNIVERSE_PERSON_INPUTS
    for variable in set(universe_owner_inputs) - {
        "age",
        "person_tax_unit_id",
        support_clone_index_column("person"),
        support_source_id_column("person"),
    }:
        register(
            "derive",
            "with_us_qbi_input_reconciliation",
            "person",
            variable,
            execution_scope="whole_pool",
            provision=universe_owner_inputs[variable],
            available_by="assembled",
        )
    register(
        "derive",
        "with_us_qbi_input_reconciliation",
        "person",
        support_source_id_column("person"),
        execution_scope="whole_pool",
        provision="assembly_support_source_identity",
        available_by="assembled",
        fallback="person_id_for_unstacked_lineage_digest",
    )
    register(
        "derive",
        "with_us_qbi_input_reconciliation",
        "person",
        "person_id",
        execution_scope="whole_pool",
        provision="frame_entity_id",
        available_by="assembled",
    )

    transfer_owned = {
        variable
        for families in pool_transfer_target_families().values()
        for variables in families.values()
        for variable in variables
    }
    for variable, entity, populace_treatment in program_bindings:
        if populace_treatment == "seed":
            provision = "administrative_seed_or_preserved_input"
            fallback = "sourced_seed_when_input_is_missing"
        elif variable in transfer_owned:
            provision = "transferred_or_preserved_input"
            fallback = None
        else:
            provision = "preserved_input_or_disclosed_engine_default"
            fallback = "checked_take_up_contract_engine_default"
        register(
            "seed",
            "seed_multispine_pool_inputs",
            entity,
            variable,
            execution_scope="whole_pool",
            provision=provision,
            available_by=(
                "transferred" if variable in transfer_owned else "seeded"
            ),
            fallback=fallback,
        )

    # Stable Bernoulli draws consume these structural columns and resolved
    # weights.  The source-identity triplet is optional as a unit: support or
    # entity identity remains the declared deterministic fallback.
    for entity in (
        "person",
        "household",
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    ):
        register(
            "seed",
            "with_us_take_up_inputs",
            entity,
            support_source_id_column(entity),
            execution_scope="whole_pool",
            provision="assembly_support_source_identity",
            available_by="assembled",
        )
    for entity in ("tax_unit", "spm_unit"):
        register(
            "seed",
            "with_us_take_up_inputs",
            entity,
            f"{entity}_id",
            execution_scope="whole_pool",
            provision="frame_entity_id",
            available_by="assembled",
        )
        register(
            "seed",
            "with_us_take_up_inputs",
            "person",
            f"person_{entity}_id",
            execution_scope="whole_pool",
            provision="frame_membership",
            available_by="assembled",
        )
        register(
            "seed",
            "with_us_take_up_inputs",
            entity,
            "<resolved_weight>",
            execution_scope="whole_pool",
            provision="frame_resolve_weights_from_household_weight",
            available_by="assembled",
        )
    for variable in ("source_year", "source_household_id", "source_person_id"):
        register(
            "seed",
            "with_us_take_up_inputs",
            "person",
            variable,
            execution_scope="whole_pool",
            provision="optional_assembled_source_identity",
            available_by="assembled",
            fallback="support_source_id_then_entity_id",
        )
    register(
        "seed",
        "with_us_take_up_inputs",
        "person",
        "age",
        execution_scope="whole_pool",
        provision="assembled_native_person_input",
        available_by="assembled",
    )

    resolved_metadata_index = (
        metadata_index
        if metadata_index is not None
        else PolicyEngineUSVariableMetadataIndex()
    )
    closure = pool_ssi_dependency_closure(resolved_metadata_index)
    take_up_variables = {variable for variable, _entity, _treatment in program_bindings}
    actual_surface_provenance = {
        "pool_transfer_target_families",
        "PRIMARY_QRF_TARGET_ORDER",
    }
    ssi_provisions: dict[str, int] = {}
    for variable in closure.input_leaves:
        metadata = resolved_metadata_index.variable_metadata(variable)
        declaration = surface.get(variable)
        if variable in POOL_DEFERRED_TRANSFER_INPUTS:
            provision = "declared_deferred_null_input"
            available_by = "transferred"
            fallback = "ephemeral_simulation_projection_engine_default"
        elif variable == "age":
            provision = "assembled_native_person_input"
            available_by = "assembled"
            fallback = None
        elif variable in take_up_variables:
            provision = "seed_stage_program_contract"
            available_by = "seeded"
            fallback = "seed_receipted_value_or_disclosed_engine_default"
        elif declaration is not None and actual_surface_provenance.intersection(
            declaration.provenance
        ):
            provision = "materialized_pool_input_surface"
            available_by = "transferred"
            fallback = None
        else:
            provision = "declared_absent_engine_input"
            available_by = "simulate"
            fallback = "policyengine_default_for_absent_input"
        ssi_provisions[provision] = ssi_provisions.get(provision, 0) + 1
        register(
            "simulate",
            "ssi_static_dependency_closure",
            metadata.entity,
            variable,
            execution_scope="whole_pool",
            provision=provision,
            available_by=available_by,
            fallback=fallback,
        )

    expected_ssi_provisions = {
        "assembled_native_person_input": 1,
        "materialized_pool_input_surface": 32,
        "seed_stage_program_contract": 1,
        "declared_deferred_null_input": 3,
        "declared_absent_engine_input": 18,
    }
    if ssi_provisions != expected_ssi_provisions:
        raise ValueError(
            "SSI input-leaf provisioning drifted; "
            f"expected={expected_ssi_provisions}, observed={ssi_provisions}."
        )

    for group in US_SCHEMA.group_entities:
        register(
            "simulate",
            "PolicyEngineUSEngine.materialize",
            group,
            US_SCHEMA.entity_id_column(group),
            execution_scope="whole_pool",
            provision="frame_entity_id",
            available_by="assembled",
        )
        register(
            "simulate",
            "PolicyEngineUSEngine.materialize",
            "person",
            US_SCHEMA.membership_column(group),
            execution_scope="whole_pool",
            provision="frame_membership",
            available_by="assembled",
        )
    register(
        "simulate",
        "PolicyEngineUSEngine.materialize",
        "person",
        US_SCHEMA.person_id_column,
        execution_scope="whole_pool",
        provision="frame_entity_id",
        available_by="assembled",
    )
    register(
        "simulate",
        "PolicyEngineUSEngine.materialize",
        "household",
        "<resolved_weight>",
        execution_scope="whole_pool",
        provision="frame_household_weight",
        available_by="assembled",
    )
    engine_structural_inputs = {
        (group, US_SCHEMA.entity_id_column(group)) for group in US_SCHEMA.group_entities
    } | {
        ("person", US_SCHEMA.membership_column(group))
        for group in US_SCHEMA.group_entities
    }
    native_engine_inputs = {
        (entity, variable)
        for entity, variables in _POOL_NATIVE_COMPLETE_OUTPUTS.items()
        for variable in variables
    }
    projection_provisions: dict[str, int] = {}
    for entity, variable in _pool_engine_input_projection(
        resolved_metadata_index,
        engine_version=closure.engine_version,
    ):
        fallback: str | None = (
            "ephemeral_simulation_projection_engine_default_if_present_null"
        )
        if variable in POOL_DEFERRED_TRANSFER_INPUTS:
            provision = "declared_deferred_null_input"
            available_by = "transferred"
            fallback = "ephemeral_simulation_projection_engine_default"
        elif variable in take_up_variables:
            provision = "seed_stage_program_contract"
            available_by = "seeded"
        elif variable in surface:
            provision = "materialized_pool_input_surface"
            available_by = "transferred"
        elif (entity, variable) in native_engine_inputs:
            provision = "assembled_native_engine_input"
            available_by = "assembled"
        elif (entity, variable) in engine_structural_inputs:
            provision = "frame_structural_engine_input"
            available_by = "assembled"
        elif (entity, variable) in _POOL_SIMULATION_PRESERVED_ENGINE_INPUTS:
            provision = "preserved_stacked_engine_input"
            available_by, preserved_fallback = _POOL_SIMULATION_PRESERVED_ENGINE_INPUTS[
                (entity, variable)
            ]
            if preserved_fallback is not None:
                fallback = preserved_fallback
        elif variable == "schedule_d_capital_gain_distributions":
            provision = "derived_schedule_d_input"
            available_by = "derived"
        else:
            provision = "declared_absent_engine_input"
            available_by = "simulate"
            fallback = "policyengine_default_if_absent"
        projection_provisions[provision] = projection_provisions.get(provision, 0) + 1
        register(
            "simulate",
            "_simulation_projection",
            entity,
            variable,
            execution_scope="disposable_simulation_copy",
            provision=provision,
            available_by=available_by,
            fallback=fallback,
        )

    expected_projection_provisions = {
        "materialized_pool_input_surface": 123,
        "seed_stage_program_contract": 13,
        "declared_deferred_null_input": 3,
        "assembled_native_engine_input": 5,
        "frame_structural_engine_input": 10,
        "preserved_stacked_engine_input": 4,
        "derived_schedule_d_input": 1,
        "declared_absent_engine_input": 704,
    }
    if projection_provisions != expected_projection_provisions:
        raise ValueError(
            "Simulation input-projection provisioning drifted; "
            f"expected={expected_projection_provisions}, "
            f"observed={projection_provisions}."
        )

    return tuple(sorted(entries.values()))


def pool_remaining_stage_input_manifest_receipt(
    metadata_index: PolicyEngineUSVariableMetadataIndex | None = None,
    *,
    take_up_program_bindings: tuple[tuple[str, str, str], ...] | None = None,
) -> dict[str, object]:
    """Return a content identity for the exhaustive post-transfer manifest."""

    manifest = pool_remaining_stage_input_manifest(
        metadata_index,
        take_up_program_bindings=take_up_program_bindings,
    )
    rows = [
        {
            "stage": entry.stage,
            "consumer": entry.consumer,
            "entity": entry.entity,
            "variable": entry.variable,
            "execution_scope": entry.execution_scope,
            "provision": entry.provision,
            "available_by": entry.available_by,
            "fallback": entry.fallback,
        }
        for entry in manifest
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            rows,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if manifest_sha256 != POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256:
        raise ValueError(
            "Remaining-stage input manifest drifted; refresh its static audit. "
            f"expected={POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256}, "
            f"observed={manifest_sha256}."
        )
    stage_counts = {
        stage: sum(entry.stage == stage for entry in manifest)
        for stage in ("derive", "seed", "simulate")
    }
    consumer_names = sorted({entry.consumer for entry in manifest})
    consumer_counts = {
        consumer: sum(entry.consumer == consumer for entry in manifest)
        for consumer in consumer_names
    }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "entry_count": len(manifest),
        "stage_counts": stage_counts,
        "consumer_counts": consumer_counts,
        "ssi_dependency_contract": {
            "engine_version": POOL_SSI_DEPENDENCY_CONTRACT.engine_version,
            "root": POOL_SSI_DEPENDENCY_CONTRACT.root,
            "input_leaf_count": POOL_SSI_DEPENDENCY_CONTRACT.input_leaf_count,
            "formula_node_count": POOL_SSI_DEPENDENCY_CONTRACT.formula_node_count,
            "edge_count": POOL_SSI_DEPENDENCY_CONTRACT.edge_count,
            "sha256": POOL_SSI_DEPENDENCY_CONTRACT.sha256,
        },
        "engine_input_projection_contract": {
            "engine_version": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.engine_version,
            "input_count": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.input_count,
            "default_count": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.default_count,
            "sha256": POOL_ENGINE_INPUT_PROJECTION_CONTRACT.sha256,
            "defaults_sha256": (POOL_ENGINE_INPUT_PROJECTION_CONTRACT.defaults_sha256),
        },
        "manifest_sha256": manifest_sha256,
    }
    receipt["sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def materialize_pool_deferred_transfer_inputs(frame: Frame) -> PoolStageOutput:
    """Represent pool-local source deferrals as typed, all-null input columns.

    A pre-existing column means the deferred owner has gained a real producer;
    fail closed rather than leaving a stale exclusion in the transfer plan.
    """

    if not isinstance(frame, Frame):
        raise TypeError(
            "Pool deferred-input materialization requires a Frame, got "
            f"{type(frame).__name__}."
        )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    receipts: dict[str, dict[str, object]] = {}
    for column, declaration in POOL_DEFERRED_TRANSFER_INPUTS.items():
        entity = declaration["entity"]
        if entity not in tables:
            raise ValueError(
                f"Pool deferred input {column!r} names missing entity {entity!r}."
            )
        if column in tables[entity]:
            raise ValueError(
                f"Pool deferred input {entity}.{column} already exists; retire "
                "the stale deferral and register its real producer."
            )
        tables[entity][column] = pd.Series(
            np.nan,
            index=tables[entity].index,
            dtype=declaration["physical_dtype"],
        )
        receipts[column] = {
            **declaration,
            "status": POOL_DEFERRED_TRANSFER_STATUS,
            "rows": int(len(tables[entity])),
            "null_rows": int(len(tables[entity])),
        }
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return PoolStageOutput(result, {"inputs": receipts})


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
    remaining_stage_authority: RemainingStagePhysicalAuthority | None = None,
    simulation_settings: SimulationSettings | None = None,
) -> PoolStageOutput:
    """Prepare source-derived values whose grain would be corrupted by cloning.

    This is the source-blind preparation subphase of the coarse ``clone``
    stage. It runs only after assembly and on a CPS-evidence projection with
    support-role metadata removed. Physical cloning then copies the outputs,
    including the transient prior-year wage target needed by the later PUF QRF.
    """

    supplied = (
        remaining_stage_authority is not None,
        simulation_settings is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "Compiler pre-clone preparation requires remaining-stage and "
            "simulation authorities together."
        )
    if all(supplied):
        if not isinstance(
            remaining_stage_authority,
            RemainingStagePhysicalAuthority,
        ):
            raise TypeError(
                "remaining_stage_authority must be a "
                "RemainingStagePhysicalAuthority."
            )
        if not isinstance(simulation_settings, SimulationSettings):
            raise TypeError("simulation_settings must be SimulationSettings.")
        operator_names = remaining_stage_authority.pre_clone_source_operator_order
        model_seed = simulation_settings.model_seed
        target_period = simulation_settings.target_period
        operators: Mapping[str, SourceFrameOperator] = {
            "derive_us_cps_carried_inputs": derive_us_cps_carried_inputs,
            "with_us_hours_worked_inputs": lambda current: (
                _with_gated_us_hours_worked_inputs(
                    current,
                    seed=model_seed,
                    time_period=target_period,
                )
            ),
            "with_us_prior_year_income_inputs": lambda current: (
                with_us_prior_year_income_inputs(
                    current,
                    seed=model_seed,
                    time_period=target_period,
                )
            ),
            "with_us_relationship_inputs": lambda current: (
                with_us_relationship_inputs(
                    current,
                    seed=model_seed,
                    time_period=target_period,
                )
            ),
            "with_us_housing_inputs": lambda current: with_us_housing_inputs(
                current,
                seed=model_seed,
                time_period=target_period,
                acs_rent_donor=acs_rent_donor,
            ),
            "with_us_eligibility_inputs": lambda current: (
                with_us_eligibility_inputs(
                    current,
                    seed=model_seed,
                    time_period=target_period,
                )
            ),
        }
        missing_operators = tuple(
            name for name in operators if name not in operator_names
        )
        unsupported_operators = tuple(
            name for name in operator_names if name not in operators
        )
        if (
            missing_operators
            or unsupported_operators
            or len(operator_names) != len(set(operator_names))
        ):
            raise ValueError(
                "Compiler pre-clone operator order must exactly cover its "
                "supported kernels; "
                f"missing={missing_operators}, "
                f"unsupported={unsupported_operators}."
            )
        return _run_source_operator_chain(
            frame,
            phase=_PRE_CLONE_PHASE,
            operator_names=operator_names,
            operators={name: operators[name] for name in operator_names},
        )

    operators = {
        "derive_us_cps_carried_inputs": derive_us_cps_carried_inputs,
        "with_us_hours_worked_inputs": _with_gated_us_hours_worked_inputs,
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


def _with_gated_us_hours_worked_inputs(
    frame: Frame,
    *,
    seed: int | None = None,
    time_period: int | None = None,
) -> PoolStageOutput:
    """Run the shared hours kernel, then keep only pool-owned input leaves."""

    resolved_seed = POOL_RANDOM_SEED if seed is None else seed
    resolved_time_period = POOL_TIME_PERIOD if time_period is None else time_period
    produced = with_us_hours_worked_inputs(
        frame,
        seed=resolved_seed,
        time_period=resolved_time_period,
    )
    gate = us_hours_worked_signal_gate(produced)
    if not gate.passed:
        raise ValueError(
            "Pool pre-clone hours-worked signal gate failed:\n  "
            + "\n  ".join(gate.failures)
        )
    pool_surface, removed = _drop_source_output_columns(
        produced,
        {"person": US_HOURS_WORKED_POOL_EXCLUDED_COLUMNS},
    )
    return PoolStageOutput(
        pool_surface,
        {
            "hours_worked_signal_gate": {
                "name": gate.name,
                "passed": True,
                "failures": [],
                "details": dict(gate.details),
            },
            "pool_excluded_outputs_removed": removed,
        },
    )


def complete_multispine_source_inputs(
    frame: Frame,
    *,
    rng: SourceRNGCapability | None = None,
) -> PoolStageOutput:
    """Run the legacy post-clone source chain through the narrow public API.

    This compatibility entrypoint retains the historical source-only order for
    callers whose late inputs are already complete. Production late-stage
    orchestration invokes :func:`run_multispine_post_clone_source_operator`
    according to the declared producer DAG, then calls
    :func:`finalize_multispine_source_inputs` once all sixteen receipts exist.
    """

    current = frame
    operator_receipts: dict[str, Mapping[str, object]] = {}
    for operator_name in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER:
        completed = run_multispine_post_clone_source_operator(
            current,
            operator_name,
            rng=rng,
        )
        current = completed.frame
        operator_receipts[operator_name] = completed.receipt
    return finalize_multispine_source_inputs(
        current,
        operator_receipts=operator_receipts,
    )


def _post_clone_source_operators(
    rng: SourceRNGCapability | None = None,
) -> Mapping[str, SourceFrameOperator]:
    """Return the fixed-seed, fixed-period post-clone kernel mapping."""

    operators: Mapping[str, SourceFrameOperator] = {
        "with_us_prior_year_income_inputs": lambda current: (
            with_us_prior_year_income_inputs(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
                rng=rng,
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
            rng=rng,
        ),
        "with_us_wic_claim_input": lambda current: with_us_wic_claim_input(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            rng=rng,
        ),
        "impute_us_housing_assistance_to_puf_support": lambda current: (
            impute_us_housing_assistance_to_puf_support(
                current,
                seed=POOL_RANDOM_SEED,
                n_estimators=POOL_HOUSING_ASSISTANCE_N_ESTIMATORS,
                max_train_samples=POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES,
                rng=rng,
            )
        ),
        "with_us_child_support_inputs": lambda current: with_us_child_support_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            allow_existing_without_source=(POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE),
            rng=rng,
        ),
        "with_us_disability_benefits": lambda current: with_us_disability_benefits(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            allow_existing_without_source=(POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE),
            rng=rng,
        ),
        "with_us_workers_compensation": lambda current: with_us_workers_compensation(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            allow_existing_without_source=(POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE),
            rng=rng,
        ),
        "with_us_weeks_unemployed": lambda current: with_us_weeks_unemployed(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            asec_2023_source=None,
            rng=rng,
        ),
        "with_us_childcare_inputs": lambda current: with_us_childcare_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            allow_existing_without_source=(POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE),
            rng=rng,
        ),
        "with_us_adult_care_inputs": lambda current: with_us_adult_care_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            allow_existing_without_source=(POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE),
            rng=rng,
        ),
        "with_us_energy_subsidy_input": lambda current: with_us_energy_subsidy_input(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            allow_existing_without_source=(POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE),
            rng=rng,
        ),
        "with_us_retirement_contribution_inputs": lambda current: (
            with_us_retirement_contribution_inputs(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
                rng=rng,
            )
        ),
        "with_us_retirement_distribution_inputs": lambda current: (
            with_us_retirement_distribution_inputs(
                current,
                seed=POOL_RANDOM_SEED,
                time_period=POOL_TIME_PERIOD,
                force_puf_imputation=True,
                rng=rng,
            )
        ),
        "with_us_immigration_inputs": lambda current: with_us_immigration_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            rng=rng,
        ),
        "with_us_education_inputs": lambda current: with_us_education_inputs(
            current,
            seed=POOL_RANDOM_SEED,
            time_period=POOL_TIME_PERIOD,
            asec_education_source=None,
        ),
    }
    return operators


def run_multispine_post_clone_source_operator(
    frame: Frame,
    operator_name: str,
    *,
    rng: SourceRNGCapability | None = None,
) -> PoolStageOutput:
    """Run exactly one declared post-clone source producer.

    Separating kernel execution from orchestration lets the late-stage DAG
    interleave source producers with the transfer groups that fill their
    declared inputs. The existing phase, projection, structure, and output
    ownership checks remain centralized in the guarded source runner.
    """

    operators = _post_clone_source_operators(rng)
    if operator_name not in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER:
        raise ValueError(
            f"{operator_name!r} is not a declared post-clone source operator; "
            f"expected one of {POOL_POST_CLONE_SOURCE_OPERATOR_ORDER}."
        )
    if set(operators) != set(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER):
        raise RuntimeError(
            "Post-clone source operator mapping drifted from its declaration; "
            f"missing={sorted(set(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER) - set(operators))}, "
            f"unexpected={sorted(set(operators) - set(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER))}."
        )
    return _run_source_operator_chain(
        frame,
        phase=_POST_CLONE_PHASE,
        operator_names=(operator_name,),
        operators={operator_name: operators[operator_name]},
    )


def finalize_multispine_source_inputs(
    frame: Frame,
    *,
    operator_receipts: Mapping[str, Mapping[str, object]],
) -> PoolStageOutput:
    """Validate complete source execution and finalize its persisted surface.

    ``operator_receipts`` must map each of the sixteen declared post-clone
    source producers to its single-operator receipt. Mapping insertion order is
    retained as the actual execution order, which may be interleaved with
    transfer producers by the late-stage DAG. Formula-owned outputs are rejected
    before the three explicitly deferred SCF inputs are materialized exactly
    once.
    """

    if not isinstance(operator_receipts, Mapping):
        raise TypeError("Post-clone source operator receipts must be a mapping.")
    receipt_items = tuple(operator_receipts.items())
    normalized: list[dict[str, object]] = []
    operator_order: list[str] = []
    evidence_receipts: list[object] = []
    for receipt_index, (receipt_operator, receipt) in enumerate(receipt_items):
        if not isinstance(receipt_operator, str):
            raise TypeError(
                "Post-clone source operator receipt keys must be strings; "
                f"receipt {receipt_index} is keyed by "
                f"{type(receipt_operator).__name__}."
            )
        if not isinstance(receipt, Mapping):
            raise TypeError(
                "Post-clone source operator receipts must be mappings; "
                f"receipt {receipt_index} is {type(receipt).__name__}."
            )
        declared_order = receipt.get("operator_order")
        if (
            receipt.get("phase") != _POST_CLONE_PHASE
            or not isinstance(declared_order, (list, tuple))
            or len(declared_order) != 1
            or not isinstance(declared_order[0], str)
        ):
            raise ValueError(
                "Each post-clone source receipt must declare phase "
                f"{_POST_CLONE_PHASE!r} and exactly one operator; receipt "
                f"{receipt_index} was {dict(receipt)!r}."
            )
        operator_name = declared_order[0]
        if operator_name != receipt_operator:
            raise ValueError(
                f"Post-clone source receipt key {receipt_operator!r} is "
                f"misbound to operator {operator_name!r}."
            )
        suboperators = receipt.get("suboperators")
        if (
            not isinstance(suboperators, (list, tuple))
            or len(suboperators) != 1
            or not isinstance(suboperators[0], Mapping)
            or suboperators[0].get("operator") != operator_name
        ):
            raise ValueError(
                "Each post-clone source receipt must carry the matching single "
                f"suboperator receipt for {operator_name!r}."
            )
        operator_order.append(operator_name)
        normalized_suboperator = dict(suboperators[0])
        normalized_suboperator["order_index"] = receipt_index
        normalized.append(normalized_suboperator)
        evidence_receipts.append(receipt.get("cps_source_evidence"))

    expected = set(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER)
    observed = set(operator_order)
    if (
        len(receipt_items) != len(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER)
        or observed != expected
    ):
        raise ValueError(
            "Post-clone source finalization requires exactly "
            f"{len(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER)} one-operator receipts; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}."
        )
    if evidence_receipts and any(
        evidence != evidence_receipts[0] for evidence in evidence_receipts[1:]
    ):
        raise ValueError(
            "Post-clone source receipts disagree on the CPS source-evidence projection."
        )

    _assert_formula_owned_source_outputs_absent(frame)
    deferred = materialize_pool_deferred_transfer_inputs(frame)
    return PoolStageOutput(
        deferred.frame,
        {
            "phase": _POST_CLONE_PHASE,
            "operator_order": operator_order,
            "cps_source_evidence": (
                evidence_receipts[0] if evidence_receipts else None
            ),
            "transient_outputs_carried_through_clone": {},
            "suboperators": normalized,
            "deferred_transfer_inputs": deferred.receipt,
        },
    )


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
            f"Multispine source operators require a Frame, got {type(frame).__name__}."
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
        if name not in POOL_OPERATOR_CONTRACTS
        or phase not in POOL_OPERATOR_CONTRACTS[name].phases
    ]
    if misplaced:
        raise ValueError(
            f"Multispine source operator(s) are not declared for {phase}: {misplaced}."
        )

    _assert_source_operator_boundary(frame, phase=phase)
    current = frame
    receipts: list[dict[str, object]] = []
    for order_index, operator_name in enumerate(operator_names):
        contract = POOL_OPERATOR_CONTRACTS[operator_name]
        family = _SOURCE_OPERATOR_FAMILIES.get(operator_name, operator_name)
        if family not in output_families:
            raise ValueError(
                f"Multispine source operator {operator_name!r} has no declared "
                f"output family {family!r}."
            )
        declared_outputs = dict(output_families[family])
        if (
            phase == _POST_CLONE_PHASE
            and contract.execution_scope == _CPS_SOURCE_EXECUTION_SCOPE
        ):
            declared_outputs = _persisted_source_outputs(declared_outputs)
        if contract.execution_scope == _CPS_SOURCE_EXECUTION_SCOPE:
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
        elif contract.execution_scope == _WHOLE_POOL_EXECUTION_SCOPE:
            available = current
        else:
            raise ValueError(
                f"Multispine operator {operator_name!r} has unknown execution "
                f"scope {contract.execution_scope!r}."
            )
        before_rows = _frame_row_counts(current)
        available_rows = _frame_row_counts(available)
        kernel_outcome = operators[operator_name](available)
        kernel_receipt: Mapping[str, object] = {}
        if isinstance(kernel_outcome, PoolStageOutput):
            outcome = kernel_outcome.frame
            kernel_receipt = kernel_outcome.receipt
        else:
            outcome = kernel_outcome
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
        overlap_ownership: Mapping[str, object] | None = None
        overlap_targets = (
            set(US_LATE_EDUCATION_NOOP_TARGETS)
            if operator_name == "with_us_education_inputs"
            else set(US_LATE_RETIREMENT_SOURCE_MIRROR_TARGETS)
            if operator_name == "with_us_retirement_contribution_inputs"
            else set()
        )
        declared_person_outputs = set(
            declared_outputs.get(available.schema.person_entity, ())
        )
        available_person_columns = set(
            available.table(available.schema.person_entity).columns
        )
        overlap_passthrough_required = bool(overlap_targets) and overlap_targets <= (
            available_person_columns
        )
        if phase == _POST_CLONE_PHASE and (
            overlap_targets & declared_person_outputs or overlap_passthrough_required
        ):
            finalized_person, overlap_ownership = _finalize_source_overlap_output(
                available.table(available.schema.person_entity),
                outcome.table(outcome.schema.person_entity),
                operator_name=operator_name,
            )
            if overlap_ownership is not None:
                tables = {entity: outcome.table(entity) for entity in outcome.entities}
                tables.update({link: outcome.link(link) for link in outcome.links})
                tables[outcome.schema.person_entity] = finalized_person
                outcome = Frame(
                    tables,
                    outcome.schema,
                    {
                        entity: outcome.weights_for(entity)
                        for entity in outcome.weighted_entities
                    },
                    outcome.strata,
                    mass_log=outcome.mass_log,
                    metadata=outcome.metadata,
                )
        _assert_source_operator_structure(
            available,
            outcome,
            operator_name=operator_name,
        )
        if contract.execution_scope == _CPS_SOURCE_EXECUTION_SCOPE:
            current, merged_rows = _merge_source_operator_outputs(
                current,
                outcome,
                declared_outputs,
                operator_name=operator_name,
            )
        else:
            current = outcome
            merged_rows = output_rows
        formula_owned_removed: dict[str, list[str]] = {}
        if (
            phase == _POST_CLONE_PHASE
            and contract.execution_scope == _CPS_SOURCE_EXECUTION_SCOPE
        ):
            formula_owned = {
                entity: frozenset(
                    set(columns) & set(_FORMULA_OWNED_SOURCE_OUTPUTS.get(entity, ()))
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
                "execution_scope": contract.execution_scope,
                "pool_input_rows": before_rows,
                "operator_input_rows": available_rows,
                "cps_available_rows": (
                    available_rows
                    if contract.execution_scope == _CPS_SOURCE_EXECUTION_SCOPE
                    else None
                ),
                "operator_output_rows": output_rows,
                "merged_rows": merged_rows,
                "operator_projection": {
                    "selection": (
                        _CPS_SOURCE_EVIDENCE_COLUMN
                        if contract.execution_scope == _CPS_SOURCE_EXECUTION_SCOPE
                        else _WHOLE_POOL_EXECUTION_SCOPE
                    ),
                    "lineage_state_persisted": (
                        contract.execution_scope == _WHOLE_POOL_EXECUTION_SCOPE
                    ),
                    "support_role_metadata_exposed": phase == _POST_CLONE_PHASE,
                },
                "output_columns": {
                    entity: sorted(columns)
                    for entity, columns in declared_outputs.items()
                    if columns
                },
                "formula_owned_outputs_removed": formula_owned_removed,
                "kernel_receipt": dict(kernel_receipt),
                "overlap_ownership": (
                    dict(overlap_ownership) if overlap_ownership is not None else None
                ),
            }
        )
    uses_cps_source = any(
        POOL_OPERATOR_CONTRACTS[name].execution_scope == _CPS_SOURCE_EXECUTION_SCOPE
        for name in operator_names
    )
    return PoolStageOutput(
        current,
        {
            "phase": phase,
            "operator_order": list(operator_names),
            "cps_source_evidence": (
                {
                    "column": _CPS_SOURCE_EVIDENCE_COLUMN,
                    "person_rows": int(
                        _cps_source_evidence_mask(frame, phase=phase).sum()
                    ),
                }
                if uses_cps_source
                else None
            ),
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
    pre_clone_invalid = phase == _PRE_CLONE_PHASE and not np.all(clone_values == 0.0)
    post_clone_invalid = phase == _POST_CLONE_PHASE and (
        not np.any(clone_values == 0.0) or not np.any(clone_values > 0.0)
    )
    if invalid_numeric or pre_clone_invalid or post_clone_invalid:
        raise ValueError(
            f"Multispine {phase} source operators received incompatible clone "
            "provenance; pre-clone requires only index 0, while post-clone "
            "requires both native and positive clone indices."
        )


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
        {entity: selected.weights_for(entity) for entity in selected.weighted_entities},
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
    """Fail closed on every formula-owned name classified at the boundary."""

    remaining = {
        entity: sorted(set(columns) & set(frame.table(entity).columns))
        for entity, columns in _FORMULA_OWNED_SOURCE_OUTPUTS.items()
        if entity in frame.entities
        and set(columns).intersection(frame.table(entity).columns)
    }
    if remaining:
        raise ValueError(
            "Completed multispine source inputs retain formula-owned source "
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


def _numeric_series_byte_receipt(
    series: pd.Series,
    *,
    boundary: str,
) -> dict[str, object]:
    values = np.ascontiguousarray(series.to_numpy(copy=False))
    if values.dtype.kind not in "biufc":
        raise TypeError(
            f"{boundary} requires a physical numeric dtype, got {series.dtype!s}."
        )
    digest = hashlib.sha256()
    digest.update(str(series.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(len(series).to_bytes(8, byteorder="little", signed=False))
    digest.update(values.tobytes())
    return {
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "sha256": digest.hexdigest(),
    }


def _finalize_source_overlap_output(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    operator_name: str,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Enforce the reviewed final owner for source-callback overlap cells."""

    education_operator = "with_us_education_inputs"
    retirement_operator = "with_us_retirement_contribution_inputs"
    if operator_name not in {education_operator, retirement_operator}:
        return after, None

    person_id = "person_id"
    required_structure = {
        person_id,
        support_clone_index_column("person"),
        support_source_id_column("person"),
    }
    missing_structure = sorted(
        required_structure - set(before.columns)
        | required_structure - set(after.columns)
    )
    if missing_structure:
        raise ValueError(
            f"US late overlap ownership for {operator_name!r} requires "
            f"person columns {missing_structure}."
        )
    if before[person_id].duplicated().any() or after[person_id].duplicated().any():
        raise ValueError(
            f"US late overlap ownership for {operator_name!r} requires unique "
            "person_id values."
        )
    if set(before[person_id]) != set(after[person_id]):
        raise ValueError(
            f"US late overlap ownership for {operator_name!r} requires unchanged "
            "person_id values."
        )

    result = after.copy(deep=True)
    targets_receipt: dict[str, object] = {}
    if operator_name == education_operator:
        for target in US_LATE_EDUCATION_NOOP_TARGETS:
            if target not in before or target not in result:
                raise ValueError(
                    f"US education overlap target person.{target} is absent."
                )
            before_values = before.set_index(person_id)[target]
            after_values = result.set_index(person_id).loc[before_values.index, target]
            before_receipt = _numeric_series_byte_receipt(
                before_values,
                boundary=f"US education overlap input person.{target}",
            )
            after_receipt = _numeric_series_byte_receipt(
                after_values,
                boundary=f"US education overlap output person.{target}",
            )
            if before_receipt != after_receipt:
                raise ValueError(
                    f"US education overlap target person.{target} violated byte "
                    "identity; its callback is consume-only."
                )
            targets_receipt[f"person.{target}"] = {
                "action": "consume_only_byte_exact_noop",
                "verified_rows": int(len(after_values)),
                "byte_identity": after_receipt,
            }
    else:
        clone_column = support_clone_index_column("person")
        source_id = support_source_id_column("person")
        clone_index = pd.to_numeric(result[clone_column], errors="raise")
        clone_values = clone_index.to_numpy(dtype=np.float64)
        if not np.equal(clone_values, np.floor(clone_values)).all():
            raise ValueError(
                "US retirement overlap ownership requires integral clone roles."
            )
        clone_one = clone_index.eq(1)
        clone_two = clone_index.eq(2)
        parents = result.loc[clone_one]
        tails = result.loc[clone_two]
        if parents[source_id].duplicated().any() or tails[source_id].duplicated().any():
            raise ValueError(
                "US retirement overlap ownership requires unique source IDs "
                "within clone roles 1 and 2."
            )
        missing_parents = sorted(set(tails[source_id]) - set(parents[source_id]))
        if missing_parents:
            raise ValueError(
                "US retirement overlap ownership found clone-2 rows without "
                f"clone-1 parents: {missing_parents}."
            )
        parent_by_source = parents.set_index(source_id)
        tail_source_ids = tails[source_id]
        clone_one_before = result.loc[clone_one].copy(deep=True)
        for target in US_LATE_RETIREMENT_SOURCE_MIRROR_TARGETS:
            if target not in result:
                raise ValueError(
                    f"US retirement overlap target person.{target} is absent."
                )
            expected = parent_by_source.loc[tail_source_ids, target]
            expected.index = tails.index
            result.loc[clone_two, target] = expected.to_numpy(copy=True)
            actual = result.loc[clone_two, target]
            expected_receipt = _numeric_series_byte_receipt(
                expected,
                boundary=f"US retirement overlap parent person.{target}",
            )
            actual_receipt = _numeric_series_byte_receipt(
                actual,
                boundary=f"US retirement overlap tail person.{target}",
            )
            if expected_receipt != actual_receipt:
                raise ValueError(
                    f"US retirement overlap target person.{target} failed its "
                    "byte-exact clone-1 mirror."
                )
            targets_receipt[f"person.{target}"] = {
                "action": "byte_exact_clone_1_mirror",
                "mirrored_clone_2_rows": int(clone_two.sum()),
                "byte_identity": actual_receipt,
            }
        if not result.loc[clone_one].equals(clone_one_before):
            raise ValueError(
                "US retirement overlap finalization changed source-owned clone-1 "
                "rows while mirroring clone 2."
            )

    ownership_receipt = us_late_overlap_ownership_receipt()
    return result, {
        "passed": True,
        "operator": operator_name,
        "ownership_sha256": ownership_receipt["sha256"],
        "targets": targets_receipt,
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
        if after_ids.duplicated().any() or set(after_ids.tolist()) != set(
            before_ids.tolist()
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
            source_values = source_by_id[column]
            aligned = source_values.reindex(target_ids)
            source_is_boolean = _is_physical_boolean_series(source_values)
            if source_is_boolean:
                positions = np.flatnonzero(eligible.to_numpy())
                aligned_boolean = pd.Series(
                    pd.array(aligned, dtype="boolean"),
                    index=target.index,
                    name=column,
                )
                if column not in target:
                    target[column] = aligned_boolean
                    continue
                incumbent = target[column]
                invalid_incumbent = incumbent.dropna().map(
                    lambda value: not isinstance(value, (bool, np.bool_))
                )
                if invalid_incumbent.any():
                    offending_types = sorted(
                        {
                            f"{type(value).__module__}.{type(value).__qualname__}"
                            for value in incumbent.dropna().loc[invalid_incumbent]
                        }
                    )
                    raise TypeError(
                        f"Multispine source operator {operator_name!r} emitted "
                        f"physical booleans for {entity}.{column}, but the pool "
                        "materialized observed non-boolean values with "
                        f"dtype {incumbent.dtype!s}: {offending_types}."
                    )
                merged_boolean = pd.Series(
                    pd.array(incumbent, dtype="boolean"),
                    index=target.index,
                    name=column,
                )
                merged_boolean.iloc[positions] = aligned_boolean.iloc[positions].array
                target[column] = merged_boolean
                continue
            if column in target and pd.api.types.is_bool_dtype(target[column].dtype):
                raise TypeError(
                    f"Multispine source operator {operator_name!r} emitted "
                    f"non-boolean values for boolean-materialized "
                    f"{entity}.{column}; source dtype={source_values.dtype!s}."
                )
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


def _is_physical_boolean_series(values: pd.Series) -> bool:
    """Recognize boolean values without treating numeric 0/1 as booleans."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return True
    observed = values.dropna()
    return bool(
        len(observed)
        and observed.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    )


def _frame_row_counts(frame: Frame) -> dict[str, int]:
    return {entity: int(len(frame.table(entity))) for entity in frame.entities}


def _resolved_derive_stage_authority(
    authority: RemainingStagePhysicalAuthority | None,
) -> tuple[tuple[str, ...], Mapping[str, object]]:
    """Resolve derive scheduling without consulting two authorities at once."""

    if authority is None:
        return (
            POOL_DERIVE_OPERATOR_ORDER,
            pool_remaining_stage_input_manifest_receipt(),
        )
    if not isinstance(authority, RemainingStagePhysicalAuthority):
        raise TypeError(
            "remaining_stage_authority must be a "
            "RemainingStagePhysicalAuthority."
        )
    for operator_name in authority.derive_operator_order:
        authority.require_inputs(stage="derive", consumer=operator_name)
    checkpoint_receipt = thaw_json(authority.checkpoint_identity_receipt)
    if not isinstance(checkpoint_receipt, dict):  # pragma: no cover - frozen type
        raise TypeError("Compiler remaining-stage receipt must be an object.")
    if checkpoint_receipt.get("manifest_sha256") != authority.manifest_receipt.get(
        "manifest_sha256"
    ):
        raise ValueError(
            "Compiler remaining-stage checkpoint and manifest receipts differ."
        )
    return authority.derive_operator_order, checkpoint_receipt


def derive_multispine_pool_inputs(
    frame: Frame,
    *,
    remaining_stage_authority: RemainingStagePhysicalAuthority | None = None,
) -> PoolStageOutput:
    """Complete deterministic post-transfer inputs without reading a spine.

    Schedule D capital-gain distributions are derived once per tax unit from
    the transferred parent inputs, then carried by the first person only when
    the unit has no pre-existing values. Existing non-null values are never
    rewritten. The shared QBI reconciliation then restores its documented
    all-or-nothing identities on the imputed PUF-detail surface.
    """

    derive_operator_order, remaining_stage_manifest_receipt = (
        _resolved_derive_stage_authority(remaining_stage_authority)
    )

    def reconcile_qbi_with_receipt(input_frame: Frame) -> PoolStageOutput:
        reconciled = with_us_qbi_input_reconciliation(input_frame)
        receipt = us_qbi_reconciliation_change_receipt(input_frame, reconciled)
        validate_us_qbi_reconciliation_transition(
            input_frame,
            reconciled,
            receipt,
            boundary="multispine QBI reconciliation generation",
        )
        return PoolStageOutput(
            reconciled,
            receipt,
        )

    completed = _run_source_operator_chain(
        frame,
        phase=_POST_CLONE_PHASE,
        operator_names=derive_operator_order,
        operators={
            "_complete_schedule_d_input": _complete_schedule_d_input,
            "with_us_qbi_input_reconciliation": reconcile_qbi_with_receipt,
        },
    )
    schedule_d_receipt = completed.receipt["suboperators"][0]["kernel_receipt"]
    qbi_receipt = completed.receipt["suboperators"][1]["kernel_receipt"]
    authorized = bind_us_qbi_reconciliation_transition_authority(
        completed.frame,
        qbi_receipt,
    )
    validate_us_qbi_reconciliation_live_output(
        authorized,
        qbi_receipt,
        boundary="multispine pool derivation output",
        expected_transition_authority_sha256=qbi_receipt["sha256"],
    )
    return PoolStageOutput(
        authorized,
        {
            "phase": _POST_CLONE_PHASE,
            "operator_order": list(derive_operator_order),
            "remaining_stage_input_manifest": remaining_stage_manifest_receipt,
            "schedule_d_capital_gain_distributions": schedule_d_receipt,
            "qbi_input_reconciliation": dict(qbi_receipt),
        },
        qbi_transition_authority_sha256=qbi_receipt["sha256"],
    )


def _complete_schedule_d_input(frame: Frame) -> PoolStageOutput:
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
    return PoolStageOutput(
        completed,
        {
            "entity": "person",
            "source_grain": "tax_unit",
            "source_columns": list(source_columns),
            "preserved_nonnull_rows": int(observed.sum()),
            "filled_rows": filled_rows,
            "derived_tax_units": derived_units,
            "partially_observed_tax_units_filled_with_zero": (partially_observed_units),
            "derivation": derivation_receipt,
        },
    )


def _take_up_program_view(program: TakeUpProgramAuthority) -> TakeUpProgram:
    """Project one compiler-issued runtime row onto the legacy kernel ABI."""

    if not isinstance(program, TakeUpProgramAuthority):
        raise TypeError(
            "take_up_authority.programs must contain TakeUpProgramAuthority values."
        )
    raw = thaw_json(program.runtime_contract)
    if not isinstance(raw, dict):  # pragma: no cover - frozen type
        raise TypeError("Compiler take-up runtime program must be an object.")
    if raw.get("variable") != program.variable:
        raise ValueError(
            "Compiler take-up program variable differs from its runtime contract."
        )
    string_fields: dict[str, str] = {}
    for field_name in (
        "engine_class",
        "entity",
        "value_type",
        "populace_treatment",
    ):
        value = raw.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Compiler take-up program requires non-empty {field_name!r}."
            )
        string_fields[field_name] = value
    if "default" not in raw:
        raise ValueError("Compiler take-up program requires a default value.")
    return TakeUpProgram(
        variable=program.variable,
        engine_class=string_fields["engine_class"],
        entity=string_fields["entity"],
        value_type=string_fields["value_type"],
        default=raw["default"],
        populace_treatment=string_fields["populace_treatment"],
        raw=raw,
    )


def _resolved_seed_stage_authorities(
    *,
    remaining_stage_authority: RemainingStagePhysicalAuthority | None,
    take_up_authority: TakeUpPhysicalAuthority | None,
    simulation_settings: SimulationSettings | None,
) -> tuple[tuple[TakeUpProgram, ...], set[str], int, int]:
    """Resolve the closed take-up stage, rejecting mixed authority modes."""

    supplied = (
        remaining_stage_authority is not None,
        take_up_authority is not None,
        simulation_settings is not None,
    )
    if not any(supplied):
        contract = load_take_up_contract()
        transfer_owned = {
            column
            for families in pool_transfer_target_families().values()
            for columns in families.values()
            for column in columns
        }
        return (
            contract.programs,
            transfer_owned,
            POOL_RANDOM_SEED,
            POOL_TIME_PERIOD,
        )
    if not all(supplied):
        raise ValueError(
            "Compiler seed execution requires remaining-stage, take-up, and "
            "simulation authorities together."
        )
    if not isinstance(remaining_stage_authority, RemainingStagePhysicalAuthority):
        raise TypeError(
            "remaining_stage_authority must be a "
            "RemainingStagePhysicalAuthority."
        )
    if not isinstance(take_up_authority, TakeUpPhysicalAuthority):
        raise TypeError("take_up_authority must be a TakeUpPhysicalAuthority.")
    if not isinstance(simulation_settings, SimulationSettings):
        raise TypeError("simulation_settings must be SimulationSettings.")

    programs = tuple(
        _take_up_program_view(program) for program in take_up_authority.programs
    )
    observed_bindings = tuple(
        (program.variable, program.entity, program.populace_treatment)
        for program in programs
    )
    if observed_bindings != take_up_authority.program_bindings:
        raise ValueError(
            "Compiler take-up programs differ from their sealed program bindings."
        )

    input_rows = remaining_stage_authority.require_inputs(
        stage="seed",
        consumer="seed_multispine_pool_inputs",
    )
    input_by_variable = {row.variable: row for row in input_rows}
    if len(input_by_variable) != len(input_rows):
        raise ValueError("Compiler seed-stage input variables must be unique.")
    program_by_variable = {program.variable: program for program in programs}
    if set(input_by_variable) != set(program_by_variable):
        raise ValueError(
            "Compiler seed-stage inputs differ from the take-up program surface."
        )
    entity_mismatches = sorted(
        variable
        for variable, program in program_by_variable.items()
        if input_by_variable[variable].entity != program.entity
    )
    if entity_mismatches:
        raise ValueError(
            "Compiler seed-stage input entities differ from take-up programs: "
            f"{entity_mismatches}."
        )

    transfer_owned = {
        row.variable for row in input_rows if row.available_by == "transferred"
    }
    return (
        programs,
        transfer_owned,
        simulation_settings.model_seed,
        simulation_settings.target_period,
    )


def seed_multispine_pool_inputs(
    frame: Frame,
    *,
    engine: _PoolRulesEngine | None = None,
    remaining_stage_authority: RemainingStagePhysicalAuthority | None = None,
    take_up_authority: TakeUpPhysicalAuthority | None = None,
    simulation_settings: SimulationSettings | None = None,
) -> PoolStageOutput:
    """Seed sourced flags, then disclose and fill unresolved engine defaults.

    TANF and EITC use their checked-in administrative seed mechanisms over the
    whole assembled pool. Existing non-null values survive byte-for-byte.
    Other take-up owners are not fabricated here: any unresolved cells receive
    the installed engine's declared default and the receipt names that fact,
    the contract treatment, and its scope owner/follow-up evidence.
    """

    programs, transfer_owned, model_seed, target_period = (
        _resolved_seed_stage_authorities(
            remaining_stage_authority=remaining_stage_authority,
            take_up_authority=take_up_authority,
            simulation_settings=simulation_settings,
        )
    )
    before = _take_up_snapshots(frame, programs)
    seeded = with_us_take_up_inputs(
        frame,
        seed=model_seed,
        time_period=target_period,
        programs=tuple(program for program in programs if program.is_seeded),
    )
    _assert_take_up_values_preserved(before, seeded)

    rules_engine = engine
    if rules_engine is None:
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        rules_engine = PolicyEngineUSEngine()
    names = [program.variable for program in programs]
    defaults = dict(rules_engine.default_values(names))

    tables = {entity: seeded.table(entity).copy() for entity in seeded.entities}
    program_receipts: dict[str, dict[str, object]] = {}
    for program in programs:
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
        program_receipts[program.variable] = {
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
            "seed": model_seed,
            "time_period": target_period,
            "programs": program_receipts,
        },
    )


def materialize_multispine_agreement_outputs(
    frame: Frame,
    *,
    engine: _PoolRulesEngine | None = None,
    simulation_settings: SimulationSettings | None = None,
) -> PoolStageOutput:
    """Materialize SSI in fixed household batches on an ephemeral gate view.

    The returned frame preserves the assembly receipt and adds ``person.ssi``.
    The caller must gate this view and publish :attr:`MultispinePoolResult.frame`
    instead; persisting ``ssi`` would pin a formula-owned output and mask
    reforms.
    """

    if simulation_settings is None:
        target_period = POOL_TIME_PERIOD
        household_batch_size = POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE
    elif not isinstance(simulation_settings, SimulationSettings):
        raise TypeError("simulation_settings must be SimulationSettings.")
    else:
        target_period = simulation_settings.target_period
        household_batch_size = simulation_settings.household_batch_size
        if household_batch_size < 1:
            raise ValueError(
                "Compiler simulation household batch size must be positive."
            )

    if any("ssi" in frame.table(entity) for entity in frame.entities):
        raise ValueError(
            "Multispine agreement simulation refuses a persisted 'ssi' column; "
            "SSI must remain formula-owned and gate-view-only."
        )
    rules_engine = engine
    if rules_engine is None:
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        rules_engine = PolicyEngineUSEngine()
        projection_contract_receipt: Mapping[str, object] = (
            pool_engine_input_projection_receipt(rules_engine)
        )
    else:
        projection_contract_receipt = {"status": "injected_test_engine"}

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
    for low in range(0, len(household_ids), household_batch_size):
        selected_households = household_ids[low : low + household_batch_size]
        person_mask = membership.isin(selected_households).to_numpy()
        selected = simulation_frame.select(person_mask)
        materialized = np.asarray(
            rules_engine.materialize(
                selected,
                ["ssi"],
                target_period,
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
                    "period": target_period,
                    "rows": int(len(person)),
                }
            },
            "household_batch_size": household_batch_size,
            "batches": batch_count,
            "engine_input_projection_contract": dict(projection_contract_receipt),
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


def _checkpoint_stage_receipts(
    resume: MultispinePoolCheckpoint,
) -> dict[str, Mapping[str, object]]:
    receipts = {name: dict(receipt) for name, receipt in resume.stage_receipts.items()}
    required = {
        "assembled": frozenset(),
        "transferred": frozenset({"impute"}),
        "simulated": frozenset({"impute", "derive", "seed", "simulate"}),
    }[resume.stage]
    allowed = required | ({"clone"} if resume.stage != "assembled" else set())
    actual = frozenset(receipts)
    if not required <= actual or not actual <= allowed:
        clone_expectation = (
            " with an optional clone receipt" if resume.stage != "assembled" else ""
        )
        raise ValueError(
            f"Multispine pool {resume.stage!r} checkpoint stage receipts are "
            f"incomplete or out of scope: expected {sorted(required)}"
            f"{clone_expectation}, got {sorted(actual)}."
        )
    return receipts


def _validated_resume_checkpoint(
    resume: MultispinePoolCheckpoint,
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    if not isinstance(resume, MultispinePoolCheckpoint):
        raise TypeError(
            f"resume must be a MultispinePoolCheckpoint, got {type(resume).__name__}."
        )
    assembly_receipt = spine_assembly_receipt(
        resume.frame,
        boundary=f"multispine pool {resume.stage} checkpoint",
    )
    if assembly_receipt != dict(resume.assembly_receipt):
        raise ValueError(
            f"Multispine pool {resume.stage!r} checkpoint assembly receipt "
            "differs from its live frame provenance."
        )
    if resume.simulation_frame is not None:
        simulation_receipt = spine_assembly_receipt(
            resume.simulation_frame,
            boundary="multispine pool simulated checkpoint evaluation frame",
        )
        if simulation_receipt != assembly_receipt:
            raise ValueError(
                "Multispine pool simulated checkpoint evaluation provenance "
                "differs from its persistent frame."
            )
    receipts = _checkpoint_stage_receipts(resume)
    if resume.stage == "simulated":
        _validate_qbi_stage_receipt(
            resume.frame,
            receipts,
            boundary="multispine pool simulated checkpoint resume",
            transition_authority_sha256=(resume.qbi_transition_authority_sha256),
        )
    return assembly_receipt, receipts


def _qbi_receipt_from_stage_receipts(
    stage_receipts: Mapping[str, Mapping[str, object]],
    *,
    boundary: str,
) -> Mapping[str, object]:
    derive = stage_receipts.get("derive")
    if not isinstance(derive, Mapping):
        raise ValueError(f"{boundary}: stage receipts have no derive object.")
    if "pool_derivation" in derive:
        raise ValueError(
            f"{boundary}: legacy checkpoint used the stacked derive receipt route."
        )
    receipt = derive.get("qbi_input_reconciliation")
    if not isinstance(receipt, Mapping):
        raise ValueError(
            f"{boundary}: derive receipt has no QBI reconciliation object."
        )
    return receipt


def _validate_qbi_stage_receipt(
    frame: Frame,
    stage_receipts: Mapping[str, Mapping[str, object]],
    *,
    boundary: str,
    transition_authority_sha256: str | None,
) -> None:
    receipt = _qbi_receipt_from_stage_receipts(
        stage_receipts,
        boundary=boundary,
    )
    validate_us_qbi_reconciliation_live_output(
        frame,
        receipt,
        boundary=boundary,
        expected_transition_authority_sha256=transition_authority_sha256,
        allowed_post_reconciliation_person_columns=(
            us_qbi_post_reconciliation_person_columns(stage_receipts.get("seed"))
        ),
    )


def _emit_pool_checkpoint(
    callback: Callable[[MultispinePoolCheckpoint], None] | None,
    *,
    stage: str,
    frame: Frame,
    assembly_receipt: Mapping[str, object],
    stage_receipts: Mapping[str, Mapping[str, object]],
    simulation_frame: Frame | None = None,
    qbi_transition_authority_sha256: str | None = None,
) -> None:
    if callback is None:
        return
    if stage == "simulated":
        _validate_qbi_stage_receipt(
            frame,
            stage_receipts,
            boundary="multispine pool simulated checkpoint emission",
            transition_authority_sha256=qbi_transition_authority_sha256,
        )
    callback(
        MultispinePoolCheckpoint(
            stage=stage,
            frame=frame,
            assembly_receipt=dict(assembly_receipt),
            stage_receipts={
                name: dict(receipt) for name, receipt in stage_receipts.items()
            },
            simulation_frame=simulation_frame,
            qbi_transition_authority_sha256=(qbi_transition_authority_sha256),
        )
    )


def run_multispine_pool_path(
    asec: Frame | None,
    acs: Frame | None,
    *,
    prepare_clone: PoolOperator | None = None,
    impute: PoolOperator,
    derive: PoolOperator,
    seed: PoolOperator,
    simulate: PoolOperator,
    agreement_gate: AgreementGate | None = None,
    checkpoint: Callable[[MultispinePoolCheckpoint], None] | None = None,
    resume: MultispinePoolCheckpoint | None = None,
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
    :func:`~microcosm.build.us_runtime.spine_agreement.spine_agreement_gate`
    with the immutable pool-specific registry and fixed tolerances.

    ``checkpoint`` observes fresh expensive boundaries in
    :data:`POOL_CHECKPOINT_STAGE_ORDER`. ``resume`` starts from one such state,
    revalidates its live assembly provenance and completed-stage receipts, and
    skips only the stages already represented there. Provenance counts and the
    terminal agreement gate are always recomputed.
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
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("checkpoint must be callable when provided.")

    resume_stage: str | None
    if resume is None:
        if not isinstance(asec, Frame) or not isinstance(acs, Frame):
            raise TypeError("Fresh multispine pool builds require ASEC and ACS Frames.")
        assembled = assemble_spines(
            {"asec": asec, "acs": acs},
            household_mass_shares=POOL_HOUSEHOLD_MASS_SHARES,
            mass_anchor_channel="asec",
        )
        assembled = canonicalize_frame_string_dtypes(
            assembled,
            boundary="multispine pool assembled checkpoint",
            in_place=True,
        )
        assembly_receipt = spine_assembly_receipt(
            assembled,
            boundary="multispine pool assembly",
        )
        receipts: dict[str, Mapping[str, object]] = {}
        qbi_transition_authority_sha256: str | None = None
        resume_stage = None
        _emit_pool_checkpoint(
            checkpoint,
            stage="assembled",
            frame=assembled,
            assembly_receipt=assembly_receipt,
            stage_receipts=receipts,
        )
    else:
        assembly_receipt, receipts = _validated_resume_checkpoint(resume)
        assembled = canonicalize_frame_string_dtypes(
            resume.frame,
            boundary=f"multispine pool {resume.stage} resume",
        )
        qbi_transition_authority_sha256 = resume.qbi_transition_authority_sha256
        resume_stage = resume.stage

    if resume_stage in {None, "assembled"}:
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
        outcome = operators["impute"](current)
        if not isinstance(outcome, PoolStageOutput):
            raise TypeError(
                "Pool impute operator must return PoolStageOutput, got "
                f"{type(outcome).__name__}."
            )
        current = canonicalize_frame_string_dtypes(
            outcome.frame,
            boundary="multispine pool transferred checkpoint",
            in_place=True,
        )
        validate_assembly_provenance(
            current,
            boundary="multispine pool impute output",
        )
        receipts["impute"] = dict(outcome.receipt)
        _emit_pool_checkpoint(
            checkpoint,
            stage="transferred",
            frame=current,
            assembly_receipt=assembly_receipt,
            stage_receipts=receipts,
        )
    else:
        current = canonicalize_frame_string_dtypes(
            resume.frame,
            boundary=f"multispine pool {resume.stage} persistent resume",
        )

    if resume_stage != "simulated":
        for stage_name in ("derive", "seed"):
            outcome = operators[stage_name](current)
            if not isinstance(outcome, PoolStageOutput):
                raise TypeError(
                    f"Pool {stage_name} operator must return PoolStageOutput, got "
                    f"{type(outcome).__name__}."
                )
            current = canonicalize_frame_string_dtypes(
                outcome.frame,
                boundary=f"multispine pool {stage_name} output",
                in_place=True,
            )
            validate_assembly_provenance(
                current,
                boundary=f"multispine pool {stage_name} output",
            )
            receipts[stage_name] = dict(outcome.receipt)
            if stage_name == "derive":
                qbi_transition_authority_sha256 = (
                    outcome.qbi_transition_authority_sha256
                )

        simulated = operators["simulate"](current)
        if not isinstance(simulated, PoolStageOutput):
            raise TypeError(
                "Pool simulate operator must return PoolStageOutput, got "
                f"{type(simulated).__name__}."
            )
        simulation_frame = canonicalize_frame_string_dtypes(
            simulated.frame,
            boundary="multispine pool simulated checkpoint",
            in_place=True,
        )
        validate_assembly_provenance(
            simulation_frame,
            boundary="multispine pool simulation output",
        )
        receipts["simulate"] = dict(simulated.receipt)
        _emit_pool_checkpoint(
            checkpoint,
            stage="simulated",
            frame=current,
            assembly_receipt=assembly_receipt,
            stage_receipts=receipts,
            simulation_frame=simulation_frame,
            qbi_transition_authority_sha256=(qbi_transition_authority_sha256),
        )
    else:
        if resume.simulation_frame is None:  # pragma: no cover - dataclass validates
            raise AssertionError("Simulated resume checkpoint has no evaluation frame.")
        simulation_frame = canonicalize_frame_string_dtypes(
            resume.simulation_frame,
            boundary="multispine pool simulated evaluation resume",
        )

    counts = spine_provenance_counts(
        current,
        boundary="multispine pool pre-agreement output",
    )

    agreement = (
        spine_agreement_gate(
            simulation_frame,
            registry=POOL_SPINE_AGREEMENT_REGISTRY,
        )
        if agreement_gate is None
        else agreement_gate(simulation_frame)
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
        qbi_transition_authority_sha256=qbi_transition_authority_sha256,
    )
