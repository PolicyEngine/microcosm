"""US stacked-spine pilot: one origin-labeled spine (microcosm#578 revision).

The ratified #578 increment-2 revision removes the two-spine / two-pipeline
agreement seam instead of patching it: ASEC and a seeded ACS household sample
are assembled into ONE spine whose rows carry their origin in the
receipt-validated support-channel columns.  Survey-specific fields are then
gap-filled cross-origin with native predictors, a single PUF pass runs after
gap-fill, and by-origin statistics replace spine-vs-spine agreement.

This module is a source-spine provenance OWNER (see the reviewed allowlist in
``test_us_spine_blindness.py``): stacking, gap-fill donor routing, activation
authority, the pre-simulation completeness gate, and the by-origin battery are
exactly the surfaces that must read origin labels.  Population operators stay
spine-blind; this module selects donors and verifies activation authority so
they never have to.

Weight harmonization (the two-arm P-lineage precedent)
------------------------------------------------------
Both origins jointly represent the same population once, exactly like the
ASEC/PUF support channels in :mod:`microcosm.build.us_runtime.puf_support`
(each channel receives a declared share of the incoming mass so the population
does not double).  :func:`~microcosm.build.us_runtime.spine_assembly.assemble_spines`
implements the allocation: for a household ``i`` of arm ``s`` with incoming
weight ``w_i``,

    ``w_i' = w_i * share_s * M_anchor / M_s``

where ``M_s`` is arm ``s``'s incoming household mass and ``M_anchor`` is the
mass-anchor arm's incoming mass.  For the seeded ACS sample,
``M_acs_sample ~= fraction * M_acs_full``, so the allocation factor contains
the inverse-sampling upweighting ``1 / fraction`` automatically; the realized
per-arm scale factors are receipted rather than assumed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import pickle
import struct
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.acs_income_universe as acs_income_universe_runtime
import microcosm.build.us_runtime.acs_transfer as acs_transfer_runtime
import microcosm.build.us_runtime.multispine_pool as multispine_pool_runtime
from microcosm.build.gates import (
    FitWeightRecord,
    GateResult,
    _sealed_stacked_gate_result,
)
from microcosm.build.serialization_dtypes import canonicalize_table_string_dtypes
from microcosm.build.source_manifest import load_source_manifest
from microcosm.build.us_runtime.acs_income_universe import (
    ACS_PUMS_EARNINGS_SOURCE_COLUMNS,
    AcsPumsEarningsUniverseApplication,
    acs_pums_earnings_universe_contract_identity,
    apply_acs_pums_earnings_universe_zeros,
    resolve_acs_pums_earnings_universe,
)
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    AcsTransferResult,
    AcsTransferTargetBank,
    TargetFamilies,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.late_producer_dag import (
    ProducerContract,
    ProducerInput,
    ProducerInputColumn,
    ProducerOutput,
    run_producer_when_ready,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_DEFERRED_TRANSFER_INPUTS,
    POOL_DEFERRED_TRANSFER_STATUS,
    POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES,
    POOL_HOUSING_ASSISTANCE_N_ESTIMATORS,
    POOL_OPERATOR_CONTRACTS,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_POST_CLONE_SOURCE_PHASE,
    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_RANDOM_SEED,
    POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE,
    POOL_SPINE_AGREEMENT_REGISTRY,
    POOL_TIME_PERIOD,
    pool_post_puf_puf_producer_target_families,
    pool_post_puf_source_producer_target_families,
    pool_post_puf_transfer_target_families,
    pool_pre_clone_gap_fill_target_families,
    pool_transfer_target_families,
)
from microcosm.build.us_runtime.operator_boundary import (
    FORMULA_OWNED_SOURCE_COLUMNS,
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    PufAggregateDisaggregationSpec,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_SUPPORT_CHANNEL,
    PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
    puf_capital_gains_tail_execution_inputs_identity,
    puf_capital_gains_tail_spec_identity,
    puf_capital_gains_tail_support_contract_identity,
    puf_capital_gains_tail_terminal_support_receipt,
    resolve_puf_capital_gains_tail_execution_inputs,
    transfer_puf_capital_gains_tail,
    validate_puf_capital_gains_tail_manifest,
    validate_puf_capital_gains_tail_terminal_support_receipt,
)
from microcosm.build.us_runtime.puf_interest_components import PufE19200AgiBand
from microcosm.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
    PRIMARY_QRF_MANIFEST_FILENAME,
    PRIMARY_QRF_TARGET_ORDER,
    PRIMARY_QRF_TARGET_ORDER_SHA256,
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    primary_puf_qrf_recipient_predictor_universe_receipt,
    run_primary_puf_qrf_chain,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    PUF_CLONE_ATTACHMENT_MANIFEST_KEY,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_PUF_SUPPORT_FIT_NAME,
    bind_puf_clone_attachment_tail_descendant,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_detail_tail_bound_quantiles_identity,
    validate_puf_clone_attachment,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_CLONE_INDEX,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    us_late_overlap_ownership_receipt,
    validate_us_late_overlap_ownership_receipt,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_PRODUCER_REGISTRY,
    CANONICAL_US_LATE_PRODUCER_SCHEDULE,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT,
    US_LATE_ACS_EARNINGS_UNIVERSE_RECEIPT_INPUT,
    US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
    US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT,
    US_LATE_PRIMARY_PUF_STAGE,
    US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
    US_LATE_PRODUCER_TRANSITION_AUTHORITY_ID,
    US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY,
    US_LATE_PRODUCER_TRANSITION_AUTHORITY_VERSION,
    US_LATE_SOURCE_EXECUTION_CONFIG_INPUT,
    US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT,
    US_LATE_SOURCE_FINALIZER_STAGE,
    US_LATE_TRANSFER_MODEL_CONFIG_INPUT,
    US_LATE_TRANSFER_TARGET_BANK_INPUT,
    us_late_producer_schedule_receipt,
)
from microcosm.frame import CONSERVE_MASS, US_SCHEMA, Frame, MassChange

__all__ = [
    "ACS_STACKED_SUPPORT_CHANNEL",
    "CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY",
    "CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY",
    "CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE",
    "CANONICAL_STACKED_DECLARED_SURFACE",
    "CANONICAL_STACKED_GAP_FILL_SURFACE",
    "CANONICAL_STACKED_GAP_FILL_PLAN",
    "CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE",
    "CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE",
    "CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE",
    "DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES",
    "ORIGIN_BATTERY_METRIC_KINDS",
    "STACKED_PILOT_ACS_SAMPLE_FRACTION",
    "STACKED_PILOT_ACS_SAMPLE_SEED",
    "STACKED_SPINE_MANIFEST_KEY",
    "US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY",
    "AbsenceProof",
    "GapFillAbsenceRule",
    "GapFillDirection",
    "GapFillResult",
    "OriginBatterySpec",
    "StackedPufPassResult",
    "StackedLateProducerResult",
    "StackedPostPufTransferResult",
    "StackedSpineResult",
    "assemble_stacked_spine",
    "assert_stacked_tail_cells_preserved",
    "by_origin_battery",
    "gap_fill_stacked_spine",
    "run_stacked_puf_pass",
    "run_stacked_late_producer_dag",
    "prepare_stacked_tail_derivation",
    "sample_acs_households",
    "stacked_completeness_gate",
    "stacked_gap_fill_plan",
    "stacked_gap_fill_producer_schedule_receipt",
    "stacked_late_primary_checkpoint_input_binding",
    "stacked_late_primary_resource_receipts",
    "stacked_late_producer_resource_semantics_receipt",
    "stacked_spine_authority_receipt",
    "transfer_stacked_post_puf_inputs",
    "transfer_stacked_post_puf_group",
    "validate_stacked_late_producer_receipt",
    "validate_stacked_late_producer_transition_authority",
    "validate_stacked_post_puf_transfer_receipt",
    "validate_stacked_spine_frame",
]

ACS_STACKED_SUPPORT_CHANNEL = "acs"

#: Fixed arm shares for the pilot stack, matching the two-arm P-lineage
#: precedent (the ASEC/PUF support expansion splits incoming mass in half so
#: two arms jointly represent the population once).  Calibration remains
#: downstream.
DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES: Mapping[str, float] = {
    BASE_ASEC_SUPPORT_CHANNEL: 0.5,
    ACS_STACKED_SUPPORT_CHANNEL: 0.5,
}

STACKED_SPINE_MANIFEST_KEY = "us_stacked_spine_manifest"
_LEGACY_STACKED_SPINE_MANIFEST_VERSION = 1
_STACKED_SPINE_MANIFEST_VERSION = 4
_SUPPORTED_STACKED_SPINE_MANIFEST_VERSIONS = {
    _LEGACY_STACKED_SPINE_MANIFEST_VERSION,
    _STACKED_SPINE_MANIFEST_VERSION,
}
_EXACT_COUNT_RULE = "floor(fraction * eligible)"
_ACS_NATIVE_GQ_LINEAGE_VERSION = 1
_ACS_NATIVE_GQ_SELECTION = "TYPEHUGQ in {2,3} on sampled native ACS rows"
_MASS_RTOL = 1e-9

#: The ratified pilot stack configuration (#578 revision): a seeded 10% ACS
#: household sample enters the spine.  Scale-up beyond the pilot changes this
#: declared fraction, never an implicit default.
STACKED_PILOT_ACS_SAMPLE_FRACTION = 0.10
STACKED_PILOT_ACS_SAMPLE_SEED = 578


@dataclass(frozen=True)
class StackedSpineResult:
    """One stacked spine plus its manifest-ready stack receipt."""

    frame: Frame
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError(
                "StackedSpineResult.frame must be a Frame, got "
                f"{type(self.frame).__name__}."
            )
        if not isinstance(self.receipt, Mapping):
            raise TypeError("StackedSpineResult.receipt must be a mapping.")


def _sample_survey_households(
    source: Frame,
    *,
    fraction: float,
    seed: int,
    source_name: str,
) -> tuple[Frame, dict[str, object]]:
    """Draw one seeded, whole-household survey sample with a receipt."""

    if not isinstance(source, Frame):
        raise TypeError(f"{source_name} must be a Frame, got {type(source).__name__}.")
    if source.schema != US_SCHEMA:
        raise ValueError(
            f"{source_name} household sampling requires the US entity schema."
        )
    _validate_fraction(fraction)
    _validate_seed(seed)
    household_channel = support_channel_column("household")
    if household_channel in source.table("household").columns:
        raise ValueError(
            f"{source_name} household sampling runs before assembly; the source "
            f"frame already carries support provenance ({household_channel!r})."
        )

    household_ids = source.table("household")["household_id"].to_numpy()
    eligible = int(len(household_ids))
    incoming_mass = float(source.weights_for("household").total)
    requested = int(math.floor(fraction * eligible))
    if requested < 1:
        raise ValueError(
            f"{source_name} sample fraction {fraction!r} floors to zero households "
            f"({_EXACT_COUNT_RULE} with eligible={eligible}); the stacked "
            "spine requires at least one sampled household."
        )

    ordered_ids = np.sort(np.asarray(household_ids, copy=True))
    if requested == eligible:
        selected_ids = ordered_ids
        sampled = source
    else:
        rng = np.random.default_rng(seed)
        selected_ids = np.sort(rng.choice(ordered_ids, size=requested, replace=False))
        person_mask = (
            source.table("person")["person_household_id"].isin(selected_ids).to_numpy()
        )
        sampled = source.select(person_mask)

    realized_ids = np.sort(sampled.table("household")["household_id"].to_numpy())
    if not np.array_equal(realized_ids, selected_ids):
        raise ValueError(
            f"{source_name} household sampling realized a different household set "
            "than it selected; whole-household selection failed."
        )
    receipt: dict[str, object] = {
        "fraction": float(fraction),
        "seed": int(seed),
        "eligible_household_count": eligible,
        "requested_household_count": requested,
        "realized_household_count": int(len(realized_ids)),
        "exact_count_rule": _EXACT_COUNT_RULE,
        "selected_household_ids_sha256": _ids_sha256(selected_ids),
        "incoming_household_mass": incoming_mass,
        "sampled_household_mass": float(sampled.weights_for("household").total),
    }
    return sampled, receipt


def sample_acs_households(
    acs: Frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Draw a seeded, whole-household ACS sample with an exact-count receipt.

    The realized count follows the deterministic exact-count rule
    ``floor(fraction * eligible)``.  Selection operates on the sorted
    household-ID inventory so equal frames produce equal samples regardless of
    incidental row order, and whole lineages (every entity row of a selected
    household) enter the sample together via :meth:`Frame.select`.

    Args:
        acs: The full pre-assembly ACS source frame (US schema, household
            weights, no support provenance).
        fraction: Household sampling fraction in ``(0, 1]``.
        seed: Non-negative integer seed for the selection RNG.

    Returns:
        The sampled frame and a JSON-ready receipt with eligible, requested,
        and realized counts, the selection digest, and mass bookkeeping.

    Raises:
        TypeError: If ``acs`` is not a Frame.
        ValueError: If the schema, configuration, or realized selection
            violates the sampling contract (including a floor of zero
            households, which fails closed).
    """

    return _sample_survey_households(
        acs,
        fraction=fraction,
        seed=seed,
        source_name="ACS",
    )


def _acs_native_group_quarters_receipt(
    sampled_acs: Frame,
    assembled: Frame,
) -> dict[str, object]:
    """Bind source GQ evidence to immutable assembly-time ACS lineages."""

    household = sampled_acs.table("household")
    person = sampled_acs.table("person")
    missing_household = sorted({"household_id", "TYPEHUGQ"} - set(household))
    missing_person = sorted({"person_id", "person_household_id"} - set(person))
    if missing_household or missing_person:
        raise ValueError(
            "Stacked ACS assembly cannot bind native group-quarters lineage; "
            f"missing_household={missing_household}, missing_person={missing_person}."
        )
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    invalid = ~kind.isin((1, 2, 3))
    if invalid.any():
        raise ValueError(
            "Stacked ACS assembly requires every sampled household to carry a "
            f"TYPEHUGQ 1/2/3 source classification; found {int(invalid.sum())} "
            "invalid row(s)."
        )
    gq_households = kind.isin((2, 3))
    household_ids = np.sort(
        household.loc[gq_households, "household_id"].to_numpy(dtype=np.int64)
    )
    gq_people = person["person_household_id"].isin(household_ids)
    person_lineages = person.loc[
        gq_people, ["person_id", "person_household_id"]
    ].to_numpy(dtype=np.int64)
    if len(person_lineages):
        person_lineages = person_lineages[
            np.lexsort((person_lineages[:, 1], person_lineages[:, 0]))
        ]
    else:
        person_lineages = person_lineages.reshape(0, 2)
    linked_counts = person.loc[gq_people, "person_household_id"].value_counts()
    if len(linked_counts) != len(household_ids) or not linked_counts.eq(1).all():
        raise ValueError(
            "Stacked ACS assembly requires exactly one native person per "
            "TYPEHUGQ 2/3 group-quarters household."
        )

    assembled_household = assembled.table("household")
    assembled_person = assembled.table("person")
    household_channel = assembled_household[support_channel_column("household")].astype(
        str
    )
    household_clone = pd.to_numeric(
        assembled_household[support_clone_index_column("household")],
        errors="raise",
    ).astype("int64")
    native_acs_household = household_channel.eq(ACS_STACKED_SUPPORT_CHANNEL) & (
        household_clone.eq(0)
    )
    assembled_kind = pd.to_numeric(assembled_household["TYPEHUGQ"], errors="coerce")
    native_household_lineages = np.column_stack(
        (
            assembled_household.loc[native_acs_household, "household_id"].to_numpy(
                dtype=np.int64
            ),
            assembled_household.loc[
                native_acs_household,
                support_source_id_column("household"),
            ].to_numpy(dtype=np.int64),
            assembled_household.loc[
                native_acs_household,
                spine_source_id_column("household"),
            ].to_numpy(dtype=np.int64),
            assembled_kind.loc[native_acs_household].to_numpy(dtype=np.int64),
        )
    )
    sorted_household_lineages = native_household_lineages[
        np.lexsort(
            tuple(
                native_household_lineages[:, column]
                for column in reversed(range(native_household_lineages.shape[1]))
            )
        )
    ]

    native_household_support_by_id = pd.Series(
        assembled_household.loc[
            native_acs_household,
            support_source_id_column("household"),
        ].to_numpy(dtype=np.int64),
        index=assembled_household.loc[native_acs_household, "household_id"].to_numpy(
            dtype=np.int64
        ),
    )
    native_household_kind_by_id = pd.Series(
        assembled_kind.loc[native_acs_household].to_numpy(dtype=np.int64),
        index=assembled_household.loc[native_acs_household, "household_id"].to_numpy(
            dtype=np.int64
        ),
    )
    person_channel = assembled_person[support_channel_column("person")].astype(str)
    person_clone = pd.to_numeric(
        assembled_person[support_clone_index_column("person")], errors="raise"
    ).astype("int64")
    native_acs_person = person_channel.eq(
        ACS_STACKED_SUPPORT_CHANNEL
    ) & person_clone.eq(0)
    parent_support = assembled_person.loc[native_acs_person, "person_household_id"].map(
        native_household_support_by_id
    )
    parent_kind = assembled_person.loc[native_acs_person, "person_household_id"].map(
        native_household_kind_by_id
    )
    if parent_support.isna().any() or parent_kind.isna().any():
        raise ValueError(
            "Stacked ACS assembly cannot bind native person-to-household lineages."
        )
    native_person_lineages = np.column_stack(
        (
            assembled_person.loc[native_acs_person, "person_id"].to_numpy(
                dtype=np.int64
            ),
            assembled_person.loc[
                native_acs_person,
                support_source_id_column("person"),
            ].to_numpy(dtype=np.int64),
            assembled_person.loc[
                native_acs_person,
                spine_source_id_column("person"),
            ].to_numpy(dtype=np.int64),
            parent_support.to_numpy(dtype=np.int64),
            parent_kind.to_numpy(dtype=np.int64),
        )
    )
    sorted_person_lineages = native_person_lineages[
        np.lexsort(
            tuple(
                native_person_lineages[:, column]
                for column in reversed(range(native_person_lineages.shape[1]))
            )
        )
    ]
    return {
        "version": _ACS_NATIVE_GQ_LINEAGE_VERSION,
        "source_channel": ACS_STACKED_SUPPORT_CHANNEL,
        "selection": _ACS_NATIVE_GQ_SELECTION,
        "one_person_per_household": True,
        "household_count": int(len(household_ids)),
        "person_count": int(len(person_lineages)),
        "household_spine_source_ids_sha256": _ids_sha256(household_ids),
        "person_spine_lineages_sha256": _integer_rows_sha256(person_lineages),
        "native_household_count": int(len(native_household_lineages)),
        "native_person_count": int(len(native_person_lineages)),
        "native_household_mapping_sha256": _integer_rows_sha256(
            sorted_household_lineages
        ),
        "native_household_order_sha256": _integer_rows_sha256(
            native_household_lineages
        ),
        "native_person_mapping_sha256": _integer_rows_sha256(sorted_person_lineages),
        "native_person_order_sha256": _integer_rows_sha256(native_person_lineages),
    }


def _normalize_sampled_household_mass(
    sampled: Frame,
    *,
    target_mass: float,
    source_name: str,
) -> tuple[Frame, float]:
    """Restore one sampled survey arm to its full-source design-weight mass."""

    weights = sampled.weights_for("household")
    sampled_mass = float(weights.total)
    if not np.isfinite(sampled_mass) or sampled_mass <= 0.0:
        raise ValueError(
            f"{source_name} sampled household mass must be positive and finite."
        )
    factor = float(target_mass / sampled_mass)
    normalized_weights = weights.with_values(weights.values * factor, weights.kind)
    mass_policy: str | MassChange = (
        CONSERVE_MASS
        if np.isclose(factor, 1.0, rtol=_MASS_RTOL, atol=0.0)
        else MassChange(
            factor=factor,
            reason=(
                f"composition-preserving {source_name} survey sampling "
                "normalization to full-source household mass"
            ),
        )
    )
    normalized = sampled.with_weights(
        "household",
        normalized_weights,
        mass=mass_policy,
    )
    return normalized, factor


def assemble_stacked_spine(
    asec: Frame,
    acs: Frame,
    *,
    acs_sample_fraction: float | None = None,
    acs_sample_seed: int | None = None,
    sample_fraction: float | None = None,
    sample_seed: int | None = None,
    household_mass_shares: Mapping[str, float] | None = None,
    mass_anchor_channel: str = BASE_ASEC_SUPPORT_CHANNEL,
) -> StackedSpineResult:
    """Assemble uniformly sampled ASEC and ACS survey arms into one spine.

    Production callers provide the single ``sample_fraction`` and
    ``sample_seed`` scale-ladder controls.  The exact same fraction is applied
    independently to both survey arms at whole-household grain; each sampled
    arm is then normalized back to its full-source household mass before the
    reviewed :func:`assemble_spines` harmonization.  This preserves each arm's
    composition and prevents the anchor population from shrinking with the
    rung.  PUF donors are not accepted here and therefore remain unsampled.

    ``acs_sample_fraction``/``acs_sample_seed`` retain the reviewed version-1
    pilot contract for reproducibility only: ASEC remains full and ACS alone
    is sampled.  Supplying pilot and production controls together fails
    closed.

    Returns:
        A validated :class:`StackedSpineResult` whose receipt mirrors the
        frozen manifest as a JSON-ready mapping.
    """

    shares = (
        dict(DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES)
        if household_mass_shares is None
        else dict(household_mass_shares)
    )
    uses_pilot_controls = acs_sample_fraction is not None or acs_sample_seed is not None
    uses_production_controls = sample_fraction is not None or sample_seed is not None
    if uses_pilot_controls == uses_production_controls:
        raise ValueError(
            "Provide exactly one complete sampling control pair: production "
            "sample_fraction/sample_seed or legacy "
            "acs_sample_fraction/acs_sample_seed."
        )

    if uses_pilot_controls:
        if acs_sample_fraction is None or acs_sample_seed is None:
            raise ValueError(
                "Legacy ACS sampling requires both acs_sample_fraction and "
                "acs_sample_seed."
            )
        sampled_asec = asec
        sampled_acs, acs_sample_receipt = sample_acs_households(
            acs,
            fraction=acs_sample_fraction,
            seed=acs_sample_seed,
        )
        incoming_masses = {
            BASE_ASEC_SUPPORT_CHANNEL: float(
                sampled_asec.weights_for("household").total
            ),
            ACS_STACKED_SUPPORT_CHANNEL: float(
                acs_sample_receipt["sampled_household_mass"]
            ),
        }
        manifest_version = _LEGACY_STACKED_SPINE_MANIFEST_VERSION
        sampling_manifest: dict[str, object] = {
            "acs_sample_fraction": float(acs_sample_fraction),
            "acs_sample_seed": int(acs_sample_seed),
            "acs_sample": acs_sample_receipt,
        }
    else:
        if sample_fraction is None or sample_seed is None:
            raise ValueError(
                "Production stacked sampling requires both sample_fraction and "
                "sample_seed."
            )
        _validate_fraction(sample_fraction)
        _validate_seed(sample_seed)
        sampled_asec_raw, asec_sample_receipt = _sample_survey_households(
            asec,
            fraction=sample_fraction,
            seed=sample_seed,
            source_name="ASEC",
        )
        sampled_acs_raw, acs_sample_receipt = _sample_survey_households(
            acs,
            fraction=sample_fraction,
            seed=sample_seed,
            source_name="ACS",
        )
        sampled_asec, asec_normalization = _normalize_sampled_household_mass(
            sampled_asec_raw,
            target_mass=float(asec.weights_for("household").total),
            source_name="ASEC",
        )
        sampled_acs, acs_normalization = _normalize_sampled_household_mass(
            sampled_acs_raw,
            target_mass=float(acs.weights_for("household").total),
            source_name="ACS",
        )
        asec_sample_receipt.update(
            {
                "normalization_factor": asec_normalization,
                "normalized_household_mass": float(
                    sampled_asec.weights_for("household").total
                ),
            }
        )
        acs_sample_receipt.update(
            {
                "normalization_factor": acs_normalization,
                "normalized_household_mass": float(
                    sampled_acs.weights_for("household").total
                ),
            }
        )
        incoming_masses = {
            BASE_ASEC_SUPPORT_CHANNEL: float(
                sampled_asec.weights_for("household").total
            ),
            ACS_STACKED_SUPPORT_CHANNEL: float(
                sampled_acs.weights_for("household").total
            ),
        }
        manifest_version = _STACKED_SPINE_MANIFEST_VERSION
        sampling_manifest = {
            "sample_fraction": float(sample_fraction),
            "sample_seed": int(sample_seed),
            "survey_samples": {
                BASE_ASEC_SUPPORT_CHANNEL: asec_sample_receipt,
                ACS_STACKED_SUPPORT_CHANNEL: acs_sample_receipt,
            },
        }

    assembled = assemble_spines(
        {
            BASE_ASEC_SUPPORT_CHANNEL: sampled_asec,
            ACS_STACKED_SUPPORT_CHANNEL: sampled_acs,
        },
        household_mass_shares=shares,
        mass_anchor_channel=mass_anchor_channel,
    )
    acs_native_group_quarters = _acs_native_group_quarters_receipt(
        sampled_acs,
        assembled,
    )

    harmonization = _harmonization_receipt(
        assembled,
        shares=shares,
        anchor_mass=incoming_masses[mass_anchor_channel],
        incoming_masses=incoming_masses,
    )
    manifest: dict[str, object] = {
        "version": manifest_version,
        **sampling_manifest,
        "household_mass_shares": {
            channel: float(share) for channel, share in shares.items()
        },
        "mass_anchor_channel": mass_anchor_channel,
        "weight_harmonization": harmonization,
        "acs_native_group_quarters": acs_native_group_quarters,
    }
    # The assembly metadata is preserved in full and augmented with the stack
    # manifest; the mass history is carried unchanged from the same source.
    stacked_metadata = {**assembled.metadata, STACKED_SPINE_MANIFEST_KEY: manifest}
    stacked_mass_log = assembled.mass_log
    stacked = Frame(
        {entity: assembled.table(entity) for entity in assembled.entities},
        assembled.schema,
        {
            entity: assembled.weights_for(entity)
            for entity in assembled.weighted_entities
        },
        assembled.strata,
        mass_log=stacked_mass_log,
        metadata=stacked_metadata,
    )
    validated = validate_stacked_spine_frame(
        stacked,
        boundary="stacked spine assembly output",
    )
    return StackedSpineResult(frame=stacked, receipt=_json_ready(validated))


def _validate_survey_sample_receipt(
    frame: Frame,
    *,
    channel: str,
    fraction: float,
    seed: int,
    sample: Mapping[str, object],
    boundary: str,
    require_normalization: bool,
) -> None:
    required_keys = {
        "eligible_household_count",
        "requested_household_count",
        "realized_household_count",
        "exact_count_rule",
        "selected_household_ids_sha256",
    }
    if require_normalization:
        required_keys.update(
            {
                "incoming_household_mass",
                "sampled_household_mass",
                "normalization_factor",
                "normalized_household_mass",
            }
        )
    missing = sorted(required_keys - set(sample))
    if missing:
        raise ValueError(
            f"{boundary}: stacked spine {channel} sample receipt is missing {missing}."
        )
    if float(sample.get("fraction", float("nan"))) != fraction:
        raise ValueError(
            f"{boundary}: stacked spine {channel} sample fraction does not "
            "match the manifest control."
        )
    if sample.get("seed") != seed:
        raise ValueError(
            f"{boundary}: stacked spine {channel} sample seed does not match "
            "the manifest control."
        )
    if sample["exact_count_rule"] != _EXACT_COUNT_RULE:
        raise ValueError(
            f"{boundary}: stacked spine {channel} sample declares exact-count "
            f"rule {sample['exact_count_rule']!r}; expected {_EXACT_COUNT_RULE!r}."
        )
    eligible = int(sample["eligible_household_count"])
    requested = int(sample["requested_household_count"])
    realized = int(sample["realized_household_count"])
    if requested != int(math.floor(fraction * eligible)):
        raise ValueError(
            f"{boundary}: stacked spine {channel} requested household count "
            f"{requested} violates {_EXACT_COUNT_RULE} for fraction={fraction!r}, "
            f"eligible={eligible}."
        )
    if realized != requested:
        raise ValueError(
            f"{boundary}: stacked spine {channel} realized household count "
            f"{realized} differs from the requested count {requested}."
        )

    household = frame.table("household")
    channel_values = household[support_channel_column("household")].astype(str)
    clone_index = household[support_clone_index_column("household")]
    native = channel_values.eq(channel) & clone_index.eq(0)
    live_count = int(native.sum())
    if live_count != realized:
        raise ValueError(
            f"{boundary}: live native {channel} household count {live_count} "
            "differs from the stacked spine manifest's realized count "
            f"{realized}."
        )
    live_ids = np.sort(
        household.loc[native, spine_source_id_column("household")].to_numpy()
    )
    live_sha = _ids_sha256(live_ids)
    if live_sha != sample["selected_household_ids_sha256"]:
        raise ValueError(
            f"{boundary}: live native {channel} household lineage digest "
            f"{live_sha} differs from the stacked spine manifest's selection "
            f"digest {sample['selected_household_ids_sha256']}."
        )
    if require_normalization:
        incoming_mass = float(sample["incoming_household_mass"])
        sampled_mass = float(sample["sampled_household_mass"])
        normalization = float(sample["normalization_factor"])
        normalized_mass = float(sample["normalized_household_mass"])
        if not np.isclose(
            sampled_mass * normalization,
            normalized_mass,
            rtol=_MASS_RTOL,
            atol=0.0,
        ) or not np.isclose(
            incoming_mass,
            normalized_mass,
            rtol=_MASS_RTOL,
            atol=0.0,
        ):
            raise ValueError(
                f"{boundary}: stacked spine {channel} sample normalization "
                "does not restore the full-source household mass."
            )


def _validated_acs_native_group_quarters_masks(
    frame: Frame,
    manifest: Mapping[str, object],
    *,
    boundary: str,
) -> tuple[pd.Series, pd.Series]:
    """Prove live ACS GQ classifications against assembly-bound lineages."""

    receipt = manifest.get("acs_native_group_quarters")
    required_receipt_keys = {
        "version",
        "source_channel",
        "selection",
        "one_person_per_household",
        "household_count",
        "person_count",
        "household_spine_source_ids_sha256",
        "person_spine_lineages_sha256",
        "native_household_count",
        "native_person_count",
        "native_household_mapping_sha256",
        "native_household_order_sha256",
        "native_person_mapping_sha256",
        "native_person_order_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required_receipt_keys:
        raise ValueError(
            f"{boundary}: stacked spine native ACS group-quarters lineage "
            "receipt is absent or malformed."
        )
    if (
        receipt.get("version") != _ACS_NATIVE_GQ_LINEAGE_VERSION
        or receipt.get("source_channel") != ACS_STACKED_SUPPORT_CHANNEL
        or receipt.get("selection") != _ACS_NATIVE_GQ_SELECTION
        or receipt.get("one_person_per_household") is not True
    ):
        raise ValueError(
            f"{boundary}: stacked spine native ACS group-quarters lineage "
            "receipt declares unsupported authority."
        )
    for receipt_field in (
        "household_count",
        "person_count",
        "native_household_count",
        "native_person_count",
    ):
        value = receipt.get(receipt_field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{boundary}: stacked spine native ACS group-quarters "
                f"{receipt_field} "
                f"must be a non-negative integer, got {value!r}."
            )
    for receipt_field in (
        "household_spine_source_ids_sha256",
        "person_spine_lineages_sha256",
        "native_household_mapping_sha256",
        "native_household_order_sha256",
        "native_person_mapping_sha256",
        "native_person_order_sha256",
    ):
        value = receipt.get(receipt_field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"{boundary}: stacked spine native ACS group-quarters "
                f"{receipt_field} "
                "must be a lowercase SHA-256 digest."
            )

    household = frame.table("household")
    person = frame.table("person")
    required_household = {
        "household_id",
        "TYPEHUGQ",
        spine_source_id_column("household"),
        support_source_id_column("household"),
        support_channel_column("household"),
        support_clone_index_column("household"),
    }
    required_person = {
        "person_id",
        "person_household_id",
        spine_source_id_column("person"),
        support_source_id_column("person"),
        support_channel_column("person"),
        support_clone_index_column("person"),
    }
    missing_household = sorted(required_household - set(household))
    missing_person = sorted(required_person - set(person))
    if missing_household or missing_person:
        raise ValueError(
            f"{boundary}: live native ACS group-quarters lineage cannot be "
            f"validated; missing_household={missing_household}, "
            f"missing_person={missing_person}."
        )

    household_channel = household[support_channel_column("household")].astype(str)
    household_clone = pd.to_numeric(
        household[support_clone_index_column("household")], errors="raise"
    ).astype("int64")
    acs_household = household_channel.eq(ACS_STACKED_SUPPORT_CHANNEL)
    native_acs_household = acs_household & household_clone.eq(0)
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    invalid_kind = acs_household & ~kind.isin((1, 2, 3))
    if invalid_kind.any():
        raise ValueError(
            f"{boundary}: live ACS group-quarters lineage has "
            f"{int(invalid_kind.sum())} row(s) without TYPEHUGQ 1/2/3."
        )
    gq_household = acs_household & kind.isin((2, 3))
    native_gq_household = native_acs_household & gq_household
    native_household_ids = np.sort(
        household.loc[
            native_gq_household,
            spine_source_id_column("household"),
        ].to_numpy(dtype=np.int64)
    )
    if (
        len(native_household_ids) != receipt["household_count"]
        or _ids_sha256(native_household_ids)
        != receipt["household_spine_source_ids_sha256"]
    ):
        raise ValueError(
            f"{boundary}: live native ACS group-quarters household lineage "
            "differs from its assembly-bound count or digest."
        )

    native_support_ids = household.loc[
        native_acs_household,
        support_source_id_column("household"),
    ]
    if native_support_ids.duplicated().any():
        raise ValueError(
            f"{boundary}: live native ACS household support lineages are not unique."
        )
    native_household_lineages = np.column_stack(
        (
            household.loc[native_acs_household, "household_id"].to_numpy(
                dtype=np.int64
            ),
            native_support_ids.to_numpy(dtype=np.int64),
            household.loc[
                native_acs_household,
                spine_source_id_column("household"),
            ].to_numpy(dtype=np.int64),
            kind.loc[native_acs_household].to_numpy(dtype=np.int64),
        )
    )
    sorted_household_lineages = native_household_lineages[
        np.lexsort(
            tuple(
                native_household_lineages[:, column]
                for column in reversed(range(native_household_lineages.shape[1]))
            )
        )
    ]
    if (
        len(native_household_lineages) != receipt["native_household_count"]
        or _integer_rows_sha256(sorted_household_lineages)
        != receipt["native_household_mapping_sha256"]
        or _integer_rows_sha256(native_household_lineages)
        != receipt["native_household_order_sha256"]
    ):
        raise ValueError(
            f"{boundary}: live native ACS household support/raw/classification "
            "mapping differs from its assembly-bound digest."
        )
    native_classification = pd.Series(
        gq_household.loc[native_acs_household].to_numpy(dtype=bool),
        index=native_support_ids.to_numpy(dtype=np.int64),
    )
    native_spine_source_by_support = pd.Series(
        household.loc[
            native_acs_household,
            spine_source_id_column("household"),
        ].to_numpy(dtype=np.int64),
        index=native_support_ids.to_numpy(dtype=np.int64),
    )
    live_acs_support_ids = household.loc[
        acs_household,
        support_source_id_column("household"),
    ]
    expected_classification = live_acs_support_ids.map(native_classification)
    expected_spine_source = live_acs_support_ids.map(native_spine_source_by_support)
    if expected_classification.isna().any() or not np.array_equal(
        expected_classification.to_numpy(dtype=bool),
        gq_household.loc[acs_household].to_numpy(dtype=bool),
    ):
        raise ValueError(
            f"{boundary}: live ACS group-quarters classification differs across "
            "clone roles from its assembly-bound native lineage."
        )
    if expected_spine_source.isna().any() or not np.array_equal(
        expected_spine_source.to_numpy(dtype=np.int64),
        household.loc[
            acs_household,
            spine_source_id_column("household"),
        ].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            f"{boundary}: live ACS support/raw household lineage pairs differ "
            "from their assembly-bound native lineage."
        )
    native_pairs = set(
        zip(
            native_support_ids.to_numpy(dtype=np.int64),
            household.loc[
                native_acs_household,
                spine_source_id_column("household"),
            ].to_numpy(dtype=np.int64),
            strict=True,
        )
    )
    for clone_role in sorted(int(value) for value in household_clone.unique()):
        role = acs_household & household_clone.eq(clone_role)
        role_pairs = list(
            zip(
                household.loc[
                    role,
                    support_source_id_column("household"),
                ].to_numpy(dtype=np.int64),
                household.loc[
                    role,
                    spine_source_id_column("household"),
                ].to_numpy(dtype=np.int64),
                strict=True,
            )
        )
        if len(role_pairs) != len(set(role_pairs)):
            raise ValueError(
                f"{boundary}: live ACS household lineages are not unique in "
                f"clone role {clone_role}."
            )
        if (
            clone_role == PUF_TAX_DETAIL_CLONE_INDEX
            and PUF_CLONE_ATTACHMENT_MANIFEST_KEY not in frame.metadata
            and set(role_pairs) != native_pairs
        ):
            raise ValueError(
                f"{boundary}: unreceipted ACS clone role {clone_role} does not "
                "exactly preserve every native support/raw lineage pair."
            )
    attachment = frame.metadata.get(PUF_CLONE_ATTACHMENT_MANIFEST_KEY)
    if attachment is not None:
        if not isinstance(attachment, Mapping):
            raise ValueError(f"{boundary}: clone attachment receipt is malformed.")
        detail = household_clone.eq(PUF_TAX_DETAIL_CLONE_INDEX)
        selected_support_ids = np.sort(
            household.loc[
                detail,
                support_source_id_column("household"),
            ].to_numpy(dtype=np.int64)
        )
        if attachment.get("realized_household_count") != int(
            len(selected_support_ids)
        ) or attachment.get("selected_household_source_ids_sha256") != _ids_sha256(
            selected_support_ids
        ):
            raise ValueError(
                f"{boundary}: live clone-1 household lineages differ from the "
                "attachment-bound selection count or digest."
            )

    person_channel = person[support_channel_column("person")].astype(str)
    person_clone = pd.to_numeric(
        person[support_clone_index_column("person")], errors="raise"
    ).astype("int64")
    native_acs_person = person_channel.eq(
        ACS_STACKED_SUPPORT_CHANNEL
    ) & person_clone.eq(0)
    native_household_support_by_live_id = pd.Series(
        household.loc[
            native_acs_household,
            support_source_id_column("household"),
        ].to_numpy(dtype=np.int64),
        index=household.loc[native_acs_household, "household_id"].to_numpy(
            dtype=np.int64
        ),
    )
    native_household_kind_by_live_id = pd.Series(
        kind.loc[native_acs_household].to_numpy(dtype=np.int64),
        index=household.loc[native_acs_household, "household_id"].to_numpy(
            dtype=np.int64
        ),
    )
    native_parent_support = person.loc[native_acs_person, "person_household_id"].map(
        native_household_support_by_live_id
    )
    native_parent_kind = person.loc[native_acs_person, "person_household_id"].map(
        native_household_kind_by_live_id
    )
    if native_parent_support.isna().any() or native_parent_kind.isna().any():
        raise ValueError(
            f"{boundary}: live native ACS person-to-household lineage cannot be "
            "resolved."
        )
    native_person_mapping = np.column_stack(
        (
            person.loc[native_acs_person, "person_id"].to_numpy(dtype=np.int64),
            person.loc[
                native_acs_person,
                support_source_id_column("person"),
            ].to_numpy(dtype=np.int64),
            person.loc[
                native_acs_person,
                spine_source_id_column("person"),
            ].to_numpy(dtype=np.int64),
            native_parent_support.to_numpy(dtype=np.int64),
            native_parent_kind.to_numpy(dtype=np.int64),
        )
    )
    sorted_person_mapping = native_person_mapping[
        np.lexsort(
            tuple(
                native_person_mapping[:, column]
                for column in reversed(range(native_person_mapping.shape[1]))
            )
        )
    ]
    if (
        len(native_person_mapping) != receipt["native_person_count"]
        or _integer_rows_sha256(sorted_person_mapping)
        != receipt["native_person_mapping_sha256"]
        or _integer_rows_sha256(native_person_mapping)
        != receipt["native_person_order_sha256"]
    ):
        raise ValueError(
            f"{boundary}: live native ACS person support/raw/parent mapping "
            "differs from its assembly-bound digest."
        )

    native_person_support = person.loc[
        native_acs_person,
        support_source_id_column("person"),
    ]
    if native_person_support.duplicated().any():
        raise ValueError(
            f"{boundary}: live native ACS person support lineages are not unique."
        )
    native_person_raw_by_support = pd.Series(
        person.loc[
            native_acs_person,
            spine_source_id_column("person"),
        ].to_numpy(dtype=np.int64),
        index=native_person_support.to_numpy(dtype=np.int64),
    )
    native_person_parent_by_support = pd.Series(
        native_parent_support.to_numpy(dtype=np.int64),
        index=native_person_support.to_numpy(dtype=np.int64),
    )
    native_person_parent_kind_by_support = pd.Series(
        native_parent_kind.to_numpy(dtype=np.int64),
        index=native_person_support.to_numpy(dtype=np.int64),
    )

    acs_person = person_channel.eq(ACS_STACKED_SUPPORT_CHANNEL)
    acs_household_live_ids = household.loc[acs_household, "household_id"]
    if acs_household_live_ids.duplicated().any():
        raise ValueError(f"{boundary}: live ACS household IDs are not unique.")
    household_support_by_live_id = pd.Series(
        household.loc[
            acs_household,
            support_source_id_column("household"),
        ].to_numpy(dtype=np.int64),
        index=acs_household_live_ids.to_numpy(dtype=np.int64),
    )
    household_kind_by_live_id = pd.Series(
        kind.loc[acs_household].to_numpy(dtype=np.int64),
        index=acs_household_live_ids.to_numpy(dtype=np.int64),
    )
    household_clone_by_live_id = pd.Series(
        household_clone.loc[acs_household].to_numpy(dtype=np.int64),
        index=acs_household_live_ids.to_numpy(dtype=np.int64),
    )
    live_person_support = person.loc[
        acs_person,
        support_source_id_column("person"),
    ]
    expected_person_raw = live_person_support.map(native_person_raw_by_support)
    expected_parent_support = live_person_support.map(native_person_parent_by_support)
    expected_parent_kind = live_person_support.map(native_person_parent_kind_by_support)
    live_parent_support = person.loc[acs_person, "person_household_id"].map(
        household_support_by_live_id
    )
    live_parent_kind = person.loc[acs_person, "person_household_id"].map(
        household_kind_by_live_id
    )
    live_parent_clone = person.loc[acs_person, "person_household_id"].map(
        household_clone_by_live_id
    )
    unresolved_person_lineage = any(
        values.isna().any()
        for values in (
            expected_person_raw,
            expected_parent_support,
            expected_parent_kind,
            live_parent_support,
            live_parent_kind,
            live_parent_clone,
        )
    )
    if unresolved_person_lineage or not (
        np.array_equal(
            expected_person_raw.to_numpy(dtype=np.int64),
            person.loc[
                acs_person,
                spine_source_id_column("person"),
            ].to_numpy(dtype=np.int64),
        )
        and np.array_equal(
            expected_parent_support.to_numpy(dtype=np.int64),
            live_parent_support.to_numpy(dtype=np.int64),
        )
        and np.array_equal(
            expected_parent_kind.to_numpy(dtype=np.int64),
            live_parent_kind.to_numpy(dtype=np.int64),
        )
        and np.array_equal(
            person_clone.loc[acs_person].to_numpy(dtype=np.int64),
            live_parent_clone.to_numpy(dtype=np.int64),
        )
    ):
        raise ValueError(
            f"{boundary}: live ACS person support/raw/parent/classification "
            "lineages differ from their assembly-bound native mappings."
        )

    for clone_role in sorted(int(value) for value in household_clone.unique()):
        role_households = acs_household & household_clone.eq(clone_role)
        role_parent_support = set(
            household.loc[
                role_households,
                support_source_id_column("household"),
            ].to_numpy(dtype=np.int64)
        )
        expected_role_people = set(
            native_person_parent_by_support.index[
                native_person_parent_by_support.isin(role_parent_support)
            ].to_numpy(dtype=np.int64)
        )
        role_people = acs_person & person_clone.eq(clone_role)
        role_person_support = person.loc[
            role_people,
            support_source_id_column("person"),
        ].to_numpy(dtype=np.int64)
        if (
            len(role_person_support) != len(set(role_person_support))
            or set(role_person_support) != expected_role_people
        ):
            raise ValueError(
                f"{boundary}: live ACS clone role {clone_role} person lineages "
                "do not exactly cover its assembly-bound household selection."
            )
    native_gq_household_live_ids = household.loc[native_gq_household, "household_id"]
    native_gq_person = native_acs_person & person["person_household_id"].isin(
        native_gq_household_live_ids
    )
    household_spine_source_by_live_id = pd.Series(
        household[spine_source_id_column("household")].to_numpy(dtype=np.int64),
        index=household["household_id"].to_numpy(dtype=np.int64),
    )
    native_person_parent_sources = person.loc[
        native_gq_person, "person_household_id"
    ].map(household_spine_source_by_live_id)
    if native_person_parent_sources.isna().any():
        raise ValueError(
            f"{boundary}: live native ACS group-quarters person parent lineage "
            "cannot be resolved."
        )
    native_person_lineages = np.column_stack(
        (
            person.loc[
                native_gq_person,
                spine_source_id_column("person"),
            ].to_numpy(dtype=np.int64),
            native_person_parent_sources.to_numpy(dtype=np.int64),
        )
    )
    if len(native_person_lineages):
        native_person_lineages = native_person_lineages[
            np.lexsort((native_person_lineages[:, 1], native_person_lineages[:, 0]))
        ]
    else:
        native_person_lineages = native_person_lineages.reshape(0, 2)
    if (
        len(native_person_lineages) != receipt["person_count"]
        or _integer_rows_sha256(native_person_lineages)
        != receipt["person_spine_lineages_sha256"]
    ):
        raise ValueError(
            f"{boundary}: live native ACS group-quarters person lineage differs "
            "from its assembly-bound count or digest."
        )

    gq_household_live_ids = household.loc[gq_household, "household_id"]
    gq_person = person_channel.eq(ACS_STACKED_SUPPORT_CHANNEL) & person[
        "person_household_id"
    ].isin(gq_household_live_ids)
    linked_counts = person.loc[gq_person, "person_household_id"].value_counts()
    if len(linked_counts) != int(gq_household.sum()) or not linked_counts.eq(1).all():
        raise ValueError(
            f"{boundary}: live ACS group-quarters lineage requires exactly one "
            "linked person per household in every clone role."
        )
    return gq_household, gq_person


def _validate_stacked_clone_role_lifecycle(
    frame: Frame,
    *,
    boundary: str,
) -> None:
    """Require one exact clone-role set authorized by the live lifecycle."""

    role_sets: dict[str, set[int]] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        clone_column = support_clone_index_column(entity)
        if clone_column not in table:
            raise ValueError(
                f"{boundary}: live stacked {entity} rows lack {clone_column!r}."
            )
        numeric = pd.to_numeric(table[clone_column], errors="raise")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(
                f"{boundary}: live stacked {entity} clone roles must be integers."
            )
        role_sets[entity] = set(numeric.to_numpy(dtype=np.int64).tolist())

    household_roles = role_sets["household"]
    attachment = frame.metadata.get(PUF_CLONE_ATTACHMENT_MANIFEST_KEY)
    if attachment is None:
        if household_roles == {0}:
            expected_roles = {0}
        elif household_roles == {0, PUF_TAX_DETAIL_CLONE_INDEX}:
            validate_puf_clone_attachment(
                frame,
                boundary=f"{boundary} full clone identity",
                expected_fraction=1.0,
                # Full-clone frames intentionally carry no attachment
                # metadata, so the seed is not part of frame identity.  The
                # validator uses this value only in its returned receipt.
                expected_seed=0,
            )
            expected_roles = {0, PUF_TAX_DETAIL_CLONE_INDEX}
        else:
            raise ValueError(
                f"{boundary}: unreceipted stacked clone roles "
                f"{sorted(household_roles)} are unauthorized."
            )
    else:
        validated_attachment = validate_puf_clone_attachment(
            frame,
            boundary=f"{boundary} clone attachment",
        )
        version = validated_attachment.get("version")
        if version == 1:
            expected_roles = {0, PUF_TAX_DETAIL_CLONE_INDEX}
        elif version == 2:
            expected_roles = {0, PUF_TAX_DETAIL_CLONE_INDEX, 2}
        else:  # pragma: no cover - the attachment validator rejects this first.
            raise ValueError(
                f"{boundary}: clone attachment authorizes no known role lifecycle."
            )

    inconsistent = {
        entity: sorted(roles)
        for entity, roles in role_sets.items()
        if roles != expected_roles
    }
    if inconsistent:
        raise ValueError(
            f"{boundary}: stacked clone roles must exactly equal "
            f"{sorted(expected_roles)} in every entity; got {inconsistent}."
        )


def validate_stacked_spine_frame(
    frame: Frame,
    *,
    boundary: str,
) -> Mapping[str, object]:
    """Validate the stacked-spine manifest against the live origin labels.

    Layered on :func:`validate_assembly_provenance` (which already proves the
    live channel counts against the frozen assembly manifest), this validator
    binds the sampling identity: the manifest's fraction and seed must be
    present and typed, the realized count must satisfy the exact-count rule,
    the live native ACS household lineage must hash to the manifest's
    selection digest, and the live per-arm household masses must match the
    declared share allocation.  Any mutation of the sample, the counts, or the
    manifest fails closed with a named error.
    """

    validate_assembly_provenance(frame, boundary=boundary)
    manifest = frame.metadata.get(STACKED_SPINE_MANIFEST_KEY)
    if manifest is None:
        raise ValueError(
            f"{boundary}: stacked spine manifest {STACKED_SPINE_MANIFEST_KEY!r} "
            "is absent."
        )
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{boundary}: stacked spine manifest is malformed.")
    version = manifest.get("version")
    if version not in _SUPPORTED_STACKED_SPINE_MANIFEST_VERSIONS:
        raise ValueError(
            f"{boundary}: stacked spine manifest has unsupported version {version!r}."
        )
    assembly = frame.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    channels = tuple(assembly["channels"])
    expected_channels = (BASE_ASEC_SUPPORT_CHANNEL, ACS_STACKED_SUPPORT_CHANNEL)
    if set(channels) != set(expected_channels):
        raise ValueError(
            f"{boundary}: stacked spine requires exactly the channels "
            f"{sorted(expected_channels)}; assembly declares {sorted(channels)}."
        )
    _validate_stacked_clone_role_lifecycle(frame, boundary=boundary)

    if version == _LEGACY_STACKED_SPINE_MANIFEST_VERSION:
        fraction = manifest.get("acs_sample_fraction")
        seed = manifest.get("acs_sample_seed")
        sample = manifest.get("acs_sample")
        sample_receipts: Mapping[str, object] = {ACS_STACKED_SUPPORT_CHANNEL: sample}
    else:
        fraction = manifest.get("sample_fraction")
        seed = manifest.get("sample_seed")
        sample_receipts = manifest.get("survey_samples")
    fraction_label = (
        "acs_sample_fraction"
        if version == _LEGACY_STACKED_SPINE_MANIFEST_VERSION
        else "sample_fraction"
    )
    seed_label = (
        "acs_sample_seed"
        if version == _LEGACY_STACKED_SPINE_MANIFEST_VERSION
        else "sample_seed"
    )
    if not isinstance(fraction, float) or isinstance(fraction, bool):
        raise ValueError(
            f"{boundary}: stacked spine manifest {fraction_label} must be "
            f"a float, got {fraction!r}."
        )
    _validate_fraction(fraction, boundary=boundary)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"{boundary}: stacked spine manifest {seed_label} must be a "
            f"non-negative integer, got {seed!r}."
        )
    expected_sample_channels = (
        (ACS_STACKED_SUPPORT_CHANNEL,)
        if version == _LEGACY_STACKED_SPINE_MANIFEST_VERSION
        else expected_channels
    )
    if not isinstance(sample_receipts, Mapping) or set(sample_receipts) != set(
        expected_sample_channels
    ):
        raise ValueError(
            f"{boundary}: stacked spine sample receipts must exactly cover "
            f"{sorted(expected_sample_channels)}."
        )
    for channel in expected_sample_channels:
        sample = sample_receipts[channel]
        if not isinstance(sample, Mapping):
            raise ValueError(
                f"{boundary}: stacked spine {channel} sample receipt is absent."
            )
        _validate_survey_sample_receipt(
            frame,
            channel=channel,
            fraction=fraction,
            seed=seed,
            sample=sample,
            boundary=boundary,
            require_normalization=version == _STACKED_SPINE_MANIFEST_VERSION,
        )

    _validated_acs_native_group_quarters_masks(
        frame,
        manifest,
        boundary=boundary,
    )

    household = frame.table("household")
    channel_values = household[support_channel_column("household")].astype(str)

    shares = manifest.get("household_mass_shares")
    if not isinstance(shares, Mapping) or set(shares) != set(expected_channels):
        raise ValueError(
            f"{boundary}: stacked spine manifest household_mass_shares must "
            f"exactly cover {sorted(expected_channels)}."
        )
    total_share = float(sum(float(value) for value in shares.values()))
    if not np.isclose(total_share, 1.0, rtol=_MASS_RTOL, atol=_MASS_RTOL):
        raise ValueError(
            f"{boundary}: stacked spine household_mass_shares sum to "
            f"{total_share!r}; expected 1.0."
        )
    harmonization = manifest.get("weight_harmonization")
    if not isinstance(harmonization, Mapping):
        raise ValueError(
            f"{boundary}: stacked spine weight-harmonization receipt is absent."
        )
    mass_anchor_channel = manifest.get("mass_anchor_channel")
    if mass_anchor_channel != channels[0]:
        raise ValueError(
            f"{boundary}: stacked spine mass_anchor_channel "
            f"{mass_anchor_channel!r} differs from the assembly anchor "
            f"channel {channels[0]!r}."
        )
    live_anchor_mass = float(frame.weights_for("household").total)
    anchor_arm = harmonization.get(mass_anchor_channel)
    if not isinstance(anchor_arm, Mapping) or "incoming_mass" not in anchor_arm:
        raise ValueError(
            f"{boundary}: stacked spine weight-harmonization receipt for "
            f"anchor {mass_anchor_channel!r} is malformed."
        )
    anchor_incoming = float(anchor_arm["incoming_mass"])
    if not np.isclose(
        anchor_incoming,
        live_anchor_mass,
        rtol=_MASS_RTOL,
        atol=0.0,
    ):
        raise ValueError(
            f"{boundary}: selected anchor {mass_anchor_channel!r} incoming "
            f"mass {anchor_incoming!r} differs from live anchor mass "
            f"{live_anchor_mass!r}."
        )

    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    for channel in expected_channels:
        arm = harmonization.get(channel)
        if not isinstance(arm, Mapping) or not {
            "allocated_mass",
            "declared_allocation",
        }.issubset(arm):
            raise ValueError(
                f"{boundary}: stacked spine weight-harmonization receipt for "
                f"{channel!r} is malformed."
            )
        live_mass = float(weights[channel_values.eq(channel).to_numpy()].sum())
        if version == _STACKED_SPINE_MANIFEST_VERSION:
            normalized_sample_mass = float(
                sample_receipts[channel]["normalized_household_mass"]
            )
            if not np.isclose(
                float(arm.get("incoming_mass", float("nan"))),
                normalized_sample_mass,
                rtol=_MASS_RTOL,
                atol=0.0,
            ):
                raise ValueError(
                    f"{boundary}: stacked spine {channel} harmonization input "
                    "mass differs from its normalized survey-sample mass."
                )
        allocated = float(arm["allocated_mass"])
        declared_allocation = float(arm["declared_allocation"])
        expected_allocation = float(shares[channel]) * live_anchor_mass
        if not np.isclose(
            declared_allocation,
            expected_allocation,
            rtol=_MASS_RTOL,
            atol=0.0,
        ):
            raise ValueError(
                f"{boundary}: declared {channel!r} allocation "
                f"{declared_allocation!r} differs from share "
                f"{float(shares[channel])!r} times live anchor mass "
                f"{live_anchor_mass!r}."
            )
        if not np.isclose(live_mass, allocated, rtol=_MASS_RTOL, atol=0.0):
            raise ValueError(
                f"{boundary}: live {channel!r} household mass {live_mass!r} "
                f"drifted from the allocated arm mass {allocated!r}."
            )
    return manifest


def _harmonization_receipt(
    assembled: Frame,
    *,
    shares: Mapping[str, float],
    anchor_mass: float,
    incoming_masses: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    household = assembled.table("household")
    channel_values = household[support_channel_column("household")].astype(str)
    weights = np.asarray(assembled.weights_for("household").values, dtype=np.float64)
    receipt: dict[str, dict[str, float]] = {}
    for channel, share in shares.items():
        incoming = float(incoming_masses[channel])
        allocated = float(weights[channel_values.eq(channel).to_numpy()].sum())
        receipt[channel] = {
            "share": float(share),
            "incoming_mass": incoming,
            "allocated_mass": allocated,
            "declared_allocation": float(share) * anchor_mass,
            "scale_factor": allocated / incoming,
        }
    return receipt


def _ids_sha256(ids: np.ndarray) -> str:
    payload = json.dumps(
        [int(value) for value in np.asarray(ids).tolist()],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _integer_rows_sha256(rows: np.ndarray) -> str:
    values = np.asarray(rows, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("Lineage digest rows must be a two-dimensional array.")
    payload = json.dumps(
        [[int(value) for value in row] for row in values.tolist()],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_fraction(fraction: float, *, boundary: str | None = None) -> None:
    prefix = f"{boundary}: " if boundary else ""
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not np.isfinite(fraction)
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError(
            f"{prefix}ACS sample fraction must be a finite number in (0, 1]; "
            f"got {fraction!r}."
        )


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"ACS sample seed must be a non-negative integer; got {seed!r}."
        )


def _json_ready(value: object) -> dict[str, object]:
    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): thaw(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [thaw(nested) for nested in item]
        if isinstance(item, (set, frozenset)):
            values = [thaw(nested) for nested in item]
            return sorted(
                values,
                key=lambda nested: json.dumps(
                    nested,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
        if isinstance(item, np.generic):
            return item.item()
        return item

    if not isinstance(value, Mapping):
        raise TypeError("Stacked spine receipts must be mappings.")
    return {str(key): thaw(item) for key, item in value.items()}


# ---------------------------------------------------------------------------
# Cross-origin gap-fill (charter item 2)
# ---------------------------------------------------------------------------

_GAP_FILL_ASEC_TO_ACS = "asec_survey_to_acs"
_GAP_FILL_ASEC_HOUSING_TO_ACS = "asec_housing_to_acs"
_GAP_FILL_HOUSING_FAMILY = "housing"
_STACKED_AUTHORITY_ID = "us_stacked_spine_authority"
# v9 binds the content-hashed execution/transition-authority schema in addition
# to the import-validated producer/input DAG. Version 8 named the graph but did
# not authenticate its live input/output transition.
_STACKED_AUTHORITY_VERSION = 9
_CANONICAL_AUTHORITY_FORM = "CANONICAL"
_NONCANONICAL_AUTHORITY_FORM = "NON-CANONICAL"
_PRE_CLONE_PREPARATION_STAGE = "prepare_multispine_source_inputs_for_clone"
_POST_GAP_FILL_STAGE = "after_gap_fill_stacked_spine"
_ACS_GQ_RENT_ABSENCE_RULE_ID = "acs_native_group_quarters_without_housing_unit"
_ACS_GQ_RENT_ABSENCE_SELECTION = "acs_typehugq_2_or_3_person"
_ACS_GQ_RENT_ABSENCE_REASON = (
    "ACS TYPEHUGQ 2/3 rows have no observed housing unit; rent must remain "
    "structurally absent rather than be synthesized as zero or donor housing."
)


def _freeze_target_families(target_families: TargetFamilies) -> TargetFamilies:
    """Recursively freeze an entity/family/target declaration."""

    if not isinstance(target_families, Mapping):
        raise TypeError("Target families must be a mapping.")
    frozen_entities: dict[str, Mapping[str, tuple[str, ...]]] = {}
    for entity, families in target_families.items():
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("Target-family entity names must be non-empty strings.")
        if not isinstance(families, Mapping):
            raise TypeError(f"Target families for {entity!r} must be a mapping.")
        frozen_families: dict[str, tuple[str, ...]] = {}
        for family, targets in families.items():
            if not isinstance(family, str) or not family.strip():
                raise ValueError("Target-family names must be non-empty strings.")
            frozen_targets = tuple(targets)
            if any(
                not isinstance(target, str) or not target.strip()
                for target in frozen_targets
            ):
                raise ValueError(
                    f"Target family {entity}/{family} contains an invalid target name."
                )
            frozen_families[family] = frozen_targets
        frozen_entities[entity] = MappingProxyType(frozen_families)
    return MappingProxyType(frozen_entities)


@dataclass(frozen=True)
class GapFillAbsenceRule:
    """One digest-bound exact recipient-universe rule for structural nulls."""

    rule_id: str
    entity: str
    column: str
    selection: str
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("rule_id", self.rule_id),
            ("entity", self.entity),
            ("column", self.column),
            ("selection", self.selection),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"GapFillAbsenceRule.{label} must be a non-empty string."
                )


@dataclass(frozen=True)
class GapFillDirection:
    """One declared cross-origin fill: recipient origin <- donor origin.

    Activation authority is declared here, not inferred from nullness: the
    named recipient channel's rows are the only rows the direction may fill,
    and the named donor channel's native rows are the only donor evidence.
    The transfer machinery itself stays spine-blind; this owner-level
    declaration is what makes the run-7 silent-skip class impossible — a
    direction either fills its declared families on its declared rows or
    fails by name.
    """

    name: str
    recipient_channel: str
    donor_channel: str
    target_families: TargetFamilies
    recipient_absence_rules: tuple[GapFillAbsenceRule, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("name", self.name),
            ("recipient_channel", self.recipient_channel),
            ("donor_channel", self.donor_channel),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"GapFillDirection.{label} must be a non-empty string."
                )
        if self.recipient_channel == self.donor_channel:
            raise ValueError(
                "GapFillDirection must fill across origins; recipient and "
                f"donor are both {self.donor_channel!r}."
            )
        if not isinstance(self.target_families, Mapping) or not any(
            families for families in self.target_families.values()
        ):
            raise ValueError(
                f"GapFillDirection {self.name!r} declares no target families."
            )
        object.__setattr__(
            self,
            "target_families",
            _freeze_target_families(self.target_families),
        )
        rules = tuple(self.recipient_absence_rules)
        if any(not isinstance(rule, GapFillAbsenceRule) for rule in rules):
            raise TypeError(
                "GapFillDirection recipient_absence_rules require "
                "GapFillAbsenceRule values."
            )
        target_keys = {
            (entity, target)
            for entity, families in self.target_families.items()
            for targets in families.values()
            for target in targets
        }
        rule_keys = [(rule.entity, rule.column) for rule in rules]
        outside = sorted(set(rule_keys) - target_keys)
        duplicates = sorted(
            key for key, count in Counter(rule_keys).items() if count > 1
        )
        if outside or duplicates:
            raise ValueError(
                f"GapFillDirection {self.name!r} has invalid recipient absence "
                f"rules; outside_targets={outside}, duplicate_targets={duplicates}."
            )
        object.__setattr__(self, "recipient_absence_rules", rules)


@dataclass(frozen=True)
class _GapFillProducerRecord:
    """One channel-aware proof that a declared target exists before its check."""

    entity: str
    family: str
    target: str
    operator: str
    operator_order_index: int
    execution_scope: str
    produced_channel: str
    producer_stage: str


@dataclass(frozen=True)
class GapFillResult:
    """The gap-filled stacked spine plus per-direction receipts."""

    frame: Frame
    receipt: Mapping[str, object]
    transfer_results: Mapping[str, AcsTransferResult] = field(default_factory=dict)


def _build_stacked_gap_fill_plan(
    families: TargetFamilies,
) -> tuple[GapFillDirection, ...]:
    """Build a deeply frozen direction plan from a declared surface."""

    survey_families: dict[str, dict[str, tuple[str, ...]]] = {}
    housing_families: dict[str, dict[str, tuple[str, ...]]] = {}
    for entity, entity_families in families.items():
        for family, targets in entity_families.items():
            bucket = (
                housing_families
                if family == _GAP_FILL_HOUSING_FAMILY
                else survey_families
            )
            bucket.setdefault(entity, {})[family] = tuple(targets)
    directions: list[GapFillDirection] = []
    if survey_families:
        directions.append(
            GapFillDirection(
                name=_GAP_FILL_ASEC_TO_ACS,
                recipient_channel=ACS_STACKED_SUPPORT_CHANNEL,
                donor_channel=BASE_ASEC_SUPPORT_CHANNEL,
                target_families=survey_families,
            )
        )
    if housing_families:
        directions.append(
            GapFillDirection(
                name=_GAP_FILL_ASEC_HOUSING_TO_ACS,
                recipient_channel=ACS_STACKED_SUPPORT_CHANNEL,
                donor_channel=BASE_ASEC_SUPPORT_CHANNEL,
                target_families=housing_families,
                recipient_absence_rules=(
                    GapFillAbsenceRule(
                        rule_id=_ACS_GQ_RENT_ABSENCE_RULE_ID,
                        entity="person",
                        column="pre_subsidy_rent",
                        selection=_ACS_GQ_RENT_ABSENCE_SELECTION,
                        reason=_ACS_GQ_RENT_ABSENCE_REASON,
                    ),
                ),
            )
        )
    return tuple(directions)


ORIGIN_BATTERY_METRIC_KINDS = (
    "boolean_incidence",
    "rare_incidence",
    "monetary_sign_separated",
    "categorical_tvd",
)


@dataclass(frozen=True)
class _BatterySupportProfile:
    profile_id: str
    version: int
    min_effective_support: int


def _freeze_authority_payload(value: object) -> object:
    """Recursively freeze one JSON-shaped authority component."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_authority_payload(nested)
                for key, nested in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_authority_payload(nested) for nested in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        "Stacked authority components must contain only canonical JSON values; "
        f"got {type(value).__name__}."
    )


@dataclass(frozen=True)
class _StackedAuthority:
    """One digest-carrying, deeply immutable stacked-spine authority bundle."""

    authority_id: str
    version: int
    gap_fill_plan: tuple[GapFillDirection, ...]
    post_puf_transfer_surface: TargetFamilies
    post_puf_puf_producer_surface: TargetFamilies
    post_puf_source_producer_surface: TargetFamilies
    declared_surface: TargetFamilies
    metric_registry: Mapping[tuple[str, str, str, int], str]
    joint_metric_registry: Mapping[tuple[str, str, tuple[str, ...], int], str]
    support_profile: _BatterySupportProfile
    puf_capital_gains_tail_support_contract: Mapping[str, object]
    late_producer_schedule: Mapping[str, object]
    declared_component_sha256: Mapping[str, str]
    declared_sha256: str
    declared_form: str

    def __post_init__(self) -> None:
        if not isinstance(self.authority_id, str) or not self.authority_id.strip():
            raise ValueError("Stacked authority_id must be a non-empty string.")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("Stacked authority version must be an integer.")
        plan = tuple(self.gap_fill_plan)
        if any(not isinstance(direction, GapFillDirection) for direction in plan):
            raise TypeError("Stacked authority plans require GapFillDirection values.")
        object.__setattr__(self, "gap_fill_plan", plan)
        object.__setattr__(
            self,
            "post_puf_transfer_surface",
            _freeze_target_families(self.post_puf_transfer_surface),
        )
        object.__setattr__(
            self,
            "post_puf_puf_producer_surface",
            _freeze_target_families(self.post_puf_puf_producer_surface),
        )
        object.__setattr__(
            self,
            "post_puf_source_producer_surface",
            _freeze_target_families(self.post_puf_source_producer_surface),
        )
        object.__setattr__(
            self,
            "declared_surface",
            _freeze_target_families(self.declared_surface),
        )
        object.__setattr__(
            self,
            "metric_registry",
            _freeze_metric_registry(self.metric_registry),
        )
        object.__setattr__(
            self,
            "joint_metric_registry",
            _freeze_joint_metric_registry(self.joint_metric_registry),
        )
        if not isinstance(self.support_profile, _BatterySupportProfile):
            raise TypeError(
                "Stacked authority support_profile must be a _BatterySupportProfile."
            )
        if not isinstance(self.puf_capital_gains_tail_support_contract, Mapping):
            raise TypeError(
                "Stacked authority capital-gains-tail support contract must be "
                "a mapping."
            )
        object.__setattr__(
            self,
            "puf_capital_gains_tail_support_contract",
            _freeze_authority_payload(self.puf_capital_gains_tail_support_contract),
        )
        if not isinstance(self.late_producer_schedule, Mapping):
            raise TypeError(
                "Stacked authority late producer schedule must be a mapping."
            )
        object.__setattr__(
            self,
            "late_producer_schedule",
            _freeze_authority_payload(self.late_producer_schedule),
        )
        component_digests = dict(self.declared_component_sha256)
        if set(component_digests) != {
            "gap_fill_plan",
            "post_puf_transfer_surface",
            "declared_surface",
            "metric_registry",
            "joint_metric_registry",
            "support_profile",
            "puf_capital_gains_tail_support_contract",
            "late_producer_schedule",
        }:
            raise ValueError(
                "Stacked authority must carry every component's declared digest."
            )
        for name, digest in component_digests.items():
            _validate_sha256(digest, boundary=f"Stacked authority {name}")
        object.__setattr__(
            self,
            "declared_component_sha256",
            MappingProxyType(component_digests),
        )
        _validate_sha256(self.declared_sha256, boundary="Stacked authority")
        if self.declared_form not in {
            _CANONICAL_AUTHORITY_FORM,
            _NONCANONICAL_AUTHORITY_FORM,
        }:
            raise ValueError(f"Unknown stacked authority form {self.declared_form!r}.")


def _validate_sha256(value: object, *, boundary: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{boundary} digest must be a lowercase sha256.")


def _freeze_metric_registry(
    registry: Mapping[tuple[str, str, str, int], str],
) -> Mapping[tuple[str, str, str, int], str]:
    if not isinstance(registry, Mapping):
        raise TypeError("The origin-battery metric registry must be a mapping.")
    frozen: dict[tuple[str, str, str, int], str] = {}
    for key, metric in registry.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 4
            or any(not isinstance(value, str) or not value for value in key[:3])
            or isinstance(key[3], bool)
            or not isinstance(key[3], int)
            or key[3] < 0
        ):
            raise ValueError(f"Invalid origin-battery metric key {key!r}.")
        if metric not in ORIGIN_BATTERY_METRIC_KINDS:
            raise ValueError(
                f"Origin-battery target {_battery_target_label(key)} declares "
                f"unknown metric {metric!r}."
            )
        frozen[key] = metric
    return MappingProxyType(frozen)


def _freeze_joint_metric_registry(
    registry: Mapping[tuple[str, str, tuple[str, ...], int], str],
) -> Mapping[tuple[str, str, tuple[str, ...], int], str]:
    if not isinstance(registry, Mapping):
        raise TypeError("The joint origin-battery metric registry must be a mapping.")
    frozen: dict[tuple[str, str, tuple[str, ...], int], str] = {}
    for key, metric in registry.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 4
            or any(not isinstance(value, str) or not value for value in key[:2])
            or not isinstance(key[2], tuple)
            or len(key[2]) < 2
            or len(set(key[2])) != len(key[2])
            or any(not isinstance(column, str) or not column for column in key[2])
            or isinstance(key[3], bool)
            or not isinstance(key[3], int)
            or key[3] < 0
        ):
            raise ValueError(f"Invalid joint origin-battery metric key {key!r}.")
        if metric != "categorical_tvd":
            raise ValueError(
                "Joint origin-battery targets must declare categorical_tvd; "
                f"got {metric!r} for {key!r}."
            )
        frozen[key] = metric
    return MappingProxyType(frozen)


def _surface_target_keys(
    surface: TargetFamilies,
) -> tuple[tuple[str, str, str, int], ...]:
    return tuple(
        sorted(
            (entity, family, target, 0)
            for entity, families in surface.items()
            for family, targets in families.items()
            for target in targets
        )
    )


def _plan_target_keys(
    plan: Sequence[GapFillDirection],
) -> tuple[tuple[str, str, str, int], ...]:
    return tuple(
        sorted(
            (entity, family, target, 0)
            for direction in plan
            for entity, families in direction.target_families.items()
            for family, targets in families.items()
            for target in targets
        )
    )


def _surface_payload(surface: TargetFamilies) -> dict[str, object]:
    return {
        entity: {family: list(targets) for family, targets in families.items()}
        for entity, families in surface.items()
    }


def _plan_payload(plan: Sequence[GapFillDirection]) -> list[dict[str, object]]:
    return [
        {
            "name": direction.name,
            "recipient_channel": direction.recipient_channel,
            "donor_channel": direction.donor_channel,
            "target_families": _surface_payload(direction.target_families),
            "recipient_absence_rules": [
                {
                    "rule_id": rule.rule_id,
                    "entity": rule.entity,
                    "column": rule.column,
                    "selection": rule.selection,
                    "reason": rule.reason,
                }
                for rule in direction.recipient_absence_rules
            ],
        }
        for direction in plan
    ]


def _build_gap_fill_producer_schedule(
    surface: TargetFamilies,
) -> tuple[_GapFillProducerRecord, ...]:
    """Resolve every early target to its actual pre-clone producer contract."""

    records: list[_GapFillProducerRecord] = []
    for entity, families in surface.items():
        for family, targets in families.items():
            for target in targets:
                for operator_order_index, operator in enumerate(
                    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
                ):
                    contract = POOL_OPERATOR_CONTRACTS[operator]
                    outputs = PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[contract.family]
                    if target not in outputs.get(entity, ()):
                        continue
                    if contract.execution_scope == "cps_source":
                        produced_channel = BASE_ASEC_SUPPORT_CHANNEL
                    elif contract.execution_scope == "whole_pool":
                        produced_channel = "*"
                    else:
                        produced_channel = f"<unsupported:{contract.execution_scope}>"
                    records.append(
                        _GapFillProducerRecord(
                            entity=entity,
                            family=family,
                            target=target,
                            operator=operator,
                            operator_order_index=operator_order_index,
                            execution_scope=contract.execution_scope,
                            produced_channel=produced_channel,
                            producer_stage=_PRE_CLONE_PREPARATION_STAGE,
                        )
                    )
    return tuple(records)


def _gap_fill_activation_stage(direction_name: str) -> str:
    return f"gap_fill_stacked_spine.activation[{direction_name}]"


def _gap_fill_producer_precedence_failures(
    plan: Sequence[GapFillDirection],
    schedule: Sequence[_GapFillProducerRecord],
) -> list[str]:
    """Fail unless every direction reads a donor its producer already populated."""

    stage_order = {
        _PRE_CLONE_PREPARATION_STAGE: 0,
        **{
            _gap_fill_activation_stage(direction.name): index + 1
            for index, direction in enumerate(plan)
        },
        _POST_GAP_FILL_STAGE: len(plan) + 1,
    }
    producer_index: dict[tuple[str, str, str], list[_GapFillProducerRecord]] = {}
    for record in schedule:
        producer_index.setdefault(
            (record.entity, record.family, record.target), []
        ).append(record)

    failures: list[str] = []
    for direction in plan:
        check_stage = _gap_fill_activation_stage(direction.name)
        check_order = stage_order[check_stage]
        for entity, families in direction.target_families.items():
            for family, targets in families.items():
                for target in targets:
                    label = f"{direction.name}/{entity}/{family}/{target}"
                    records = producer_index.get((entity, family, target), [])
                    if len(records) != 1:
                        failures.append(
                            f"{label}: expected exactly one declared pre-clone "
                            f"producer, found {len(records)}."
                        )
                        continue
                    record = records[0]
                    if record.execution_scope not in {"cps_source", "whole_pool"}:
                        failures.append(
                            f"{label}: producer {record.operator!r} declares "
                            f"unknown execution scope {record.execution_scope!r}."
                        )
                        continue
                    producer_order = stage_order.get(record.producer_stage)
                    if producer_order is None:
                        failures.append(
                            f"{label}: producer {record.operator!r} declares "
                            f"unknown stage {record.producer_stage!r}."
                        )
                    elif producer_order >= check_order:
                        failures.append(
                            f"{label}: producer {record.operator!r} runs at "
                            f"{record.producer_stage!r}, which does not precede "
                            f"activation stage {check_stage!r}."
                        )
                    if record.produced_channel not in {
                        direction.donor_channel,
                        "*",
                    }:
                        failures.append(
                            f"{label}: producer {record.operator!r} populates "
                            f"channel {record.produced_channel!r}, but activation "
                            f"declares donor {direction.donor_channel!r}."
                        )
    return failures


def _metric_registry_payload(
    registry: Mapping[tuple[str, str, str, int], str],
) -> list[dict[str, object]]:
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
    registry: Mapping[tuple[str, str, tuple[str, ...], int], str],
) -> list[dict[str, object]]:
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


def _support_profile_payload(profile: _BatterySupportProfile) -> dict[str, object]:
    return {
        "min_effective_support": profile.min_effective_support,
        "profile_id": profile.profile_id,
        "version": profile.version,
    }


def _authority_component_payloads(
    *,
    gap_fill_plan: Sequence[GapFillDirection],
    post_puf_transfer_surface: TargetFamilies,
    post_puf_puf_producer_surface: TargetFamilies,
    post_puf_source_producer_surface: TargetFamilies,
    declared_surface: TargetFamilies,
    metric_registry: Mapping[tuple[str, str, str, int], str],
    joint_metric_registry: Mapping[tuple[str, str, tuple[str, ...], int], str],
    support_profile: _BatterySupportProfile,
    puf_capital_gains_tail_support_contract: Mapping[str, object],
    late_producer_schedule: Mapping[str, object],
) -> dict[str, object]:
    return {
        "gap_fill_plan": _plan_payload(gap_fill_plan),
        "post_puf_transfer_surface": {
            "donor_channel": BASE_ASEC_SUPPORT_CHANNEL,
            "donor_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
            "recipient_selection": (
                "target_specific_complement_of_declared_producer_rows"
            ),
            "producer_surfaces": {
                "puf_clone": _surface_payload(post_puf_puf_producer_surface),
                "post_clone_source": _surface_payload(post_puf_source_producer_surface),
            },
            "target_families": _surface_payload(post_puf_transfer_surface),
        },
        "declared_surface": _surface_payload(declared_surface),
        "metric_registry": _metric_registry_payload(metric_registry),
        "joint_metric_registry": _joint_metric_registry_payload(joint_metric_registry),
        "support_profile": _support_profile_payload(support_profile),
        "puf_capital_gains_tail_support_contract": _json_ready(
            puf_capital_gains_tail_support_contract
        ),
        "late_producer_schedule": _json_ready(late_producer_schedule),
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _authority_live_digests(
    authority: _StackedAuthority,
) -> tuple[dict[str, str], str]:
    payloads = _authority_component_payloads(
        gap_fill_plan=authority.gap_fill_plan,
        post_puf_transfer_surface=authority.post_puf_transfer_surface,
        post_puf_puf_producer_surface=(authority.post_puf_puf_producer_surface),
        post_puf_source_producer_surface=(authority.post_puf_source_producer_surface),
        declared_surface=authority.declared_surface,
        metric_registry=authority.metric_registry,
        joint_metric_registry=authority.joint_metric_registry,
        support_profile=authority.support_profile,
        puf_capital_gains_tail_support_contract=(
            authority.puf_capital_gains_tail_support_contract
        ),
        late_producer_schedule=authority.late_producer_schedule,
    )
    component_digests = {
        name: _canonical_sha256(payload) for name, payload in payloads.items()
    }
    bundle_digest = _canonical_sha256(
        {
            "authority_id": authority.authority_id,
            "version": authority.version,
            "components": payloads,
        }
    )
    return component_digests, bundle_digest


def _make_stacked_authority(
    *,
    authority_id: str,
    version: int,
    gap_fill_plan: Sequence[GapFillDirection],
    post_puf_transfer_surface: TargetFamilies,
    post_puf_puf_producer_surface: TargetFamilies,
    post_puf_source_producer_surface: TargetFamilies,
    declared_surface: TargetFamilies,
    metric_registry: Mapping[tuple[str, str, str, int], str],
    support_profile: _BatterySupportProfile,
    declared_form: str,
    puf_capital_gains_tail_support_contract: Mapping[str, object] | None = None,
    late_producer_schedule: Mapping[str, object] | None = None,
    joint_metric_registry: Mapping[tuple[str, str, tuple[str, ...], int], str]
    | None = None,
    declared_component_sha256: Mapping[str, str] | None = None,
    declared_sha256: str | None = None,
) -> _StackedAuthority:
    frozen_plan = tuple(gap_fill_plan)
    frozen_post_puf_surface = _freeze_target_families(post_puf_transfer_surface)
    frozen_post_puf_puf_producer_surface = _freeze_target_families(
        post_puf_puf_producer_surface
    )
    frozen_post_puf_source_producer_surface = _freeze_target_families(
        post_puf_source_producer_surface
    )
    frozen_surface = _freeze_target_families(declared_surface)
    frozen_registry = _freeze_metric_registry(metric_registry)
    frozen_joint_registry = _freeze_joint_metric_registry(
        {} if joint_metric_registry is None else joint_metric_registry
    )
    frozen_tail_support_contract = _freeze_authority_payload(
        puf_capital_gains_tail_support_contract_identity()
        if puf_capital_gains_tail_support_contract is None
        else puf_capital_gains_tail_support_contract
    )
    if not isinstance(frozen_tail_support_contract, Mapping):
        raise TypeError("Capital-gains-tail support contract must be a mapping.")
    frozen_late_producer_schedule = _freeze_authority_payload(
        us_late_producer_schedule_receipt()
        if late_producer_schedule is None
        else late_producer_schedule
    )
    if not isinstance(frozen_late_producer_schedule, Mapping):
        raise TypeError("Late producer schedule must be a mapping.")
    component_payloads = _authority_component_payloads(
        gap_fill_plan=frozen_plan,
        post_puf_transfer_surface=frozen_post_puf_surface,
        post_puf_puf_producer_surface=frozen_post_puf_puf_producer_surface,
        post_puf_source_producer_surface=frozen_post_puf_source_producer_surface,
        declared_surface=frozen_surface,
        metric_registry=frozen_registry,
        joint_metric_registry=frozen_joint_registry,
        support_profile=support_profile,
        puf_capital_gains_tail_support_contract=frozen_tail_support_contract,
        late_producer_schedule=frozen_late_producer_schedule,
    )
    live_components = {
        name: _canonical_sha256(payload) for name, payload in component_payloads.items()
    }
    live_bundle = _canonical_sha256(
        {
            "authority_id": authority_id,
            "version": version,
            "components": component_payloads,
        }
    )
    return _StackedAuthority(
        authority_id=authority_id,
        version=version,
        gap_fill_plan=frozen_plan,
        post_puf_transfer_surface=frozen_post_puf_surface,
        post_puf_puf_producer_surface=frozen_post_puf_puf_producer_surface,
        post_puf_source_producer_surface=frozen_post_puf_source_producer_surface,
        declared_surface=frozen_surface,
        metric_registry=frozen_registry,
        joint_metric_registry=frozen_joint_registry,
        support_profile=support_profile,
        puf_capital_gains_tail_support_contract=frozen_tail_support_contract,
        late_producer_schedule=frozen_late_producer_schedule,
        declared_component_sha256=(
            live_components
            if declared_component_sha256 is None
            else declared_component_sha256
        ),
        declared_sha256=live_bundle if declared_sha256 is None else declared_sha256,
        declared_form=declared_form,
    )


def _test_metric_registry_for_surface(
    surface: TargetFamilies,
) -> Mapping[tuple[str, str, str, int], str]:
    """Choose fixture metrics only; production uses explicit declarations."""

    boolean_columns = {
        "estate_income_would_be_qualified",
        "farm_operations_income_would_be_qualified",
        "farm_rent_income_would_be_qualified",
        "partnership_s_corp_income_would_be_qualified",
        "rental_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
        "sstb_self_employment_income_would_be_qualified",
        "business_is_sstb",
        "is_incapable_of_self_care",
    }
    registry: dict[tuple[str, str, str, int], str] = {}
    for key in _surface_target_keys(surface):
        _entity, family, column, _clone_index = key
        if (
            family in {"model_required_boolean", "benefit_participation"}
            or column in boolean_columns
        ):
            metric = "boolean_incidence"
        elif (
            family == "model_required_discrete"
            or column == "first_home_mortgage_origination_year"
        ):
            metric = "categorical_tvd"
        else:
            metric = "monetary_sign_separated"
        registry[key] = metric
    return MappingProxyType(registry)


def _terminal_surface_from_pool_registry() -> TargetFamilies:
    surface: dict[str, dict[str, tuple[str, ...]]] = {}
    for spec in POOL_SPINE_AGREEMENT_REGISTRY:
        surface.setdefault(spec.entity, {})[spec.family] = tuple(spec.columns)
    return _freeze_target_families(surface)


_EXPLICIT_ORIGIN_BATTERY_METRIC_DECLARATIONS: Mapping[
    str, tuple[tuple[str, str, str], ...]
] = MappingProxyType(
    {
        "monetary_sign_separated": (
            ("person", "adult_care", "pre_subsidy_care_expenses"),
            (
                "person",
                "derived_transfer",
                "schedule_d_capital_gain_distributions",
            ),
            ("person", "housing", "pre_subsidy_rent"),
            (
                "person",
                "model_required_numeric",
                "health_insurance_premiums_without_medicare_part_b",
            ),
            ("person", "model_required_numeric", "hours_worked_last_week"),
            ("person", "model_required_numeric", "other_medical_expenses"),
            (
                "person",
                "model_required_numeric",
                "over_the_counter_health_expenses",
            ),
            (
                "person",
                "model_required_numeric",
                "tax_exempt_private_pension_income",
            ),
            (
                "person",
                "model_required_numeric",
                "unemployment_compensation",
            ),
            ("person", "model_required_numeric", "veterans_benefits"),
            ("person", "puf_tax_itemization", "taxable_interest_income"),
            ("person", "puf_tax_itemization", "qualified_dividend_income"),
            ("person", "puf_tax_itemization", "non_qualified_dividend_income"),
            ("person", "puf_tax_itemization", "tax_exempt_interest_income"),
            ("person", "puf_tax_itemization", "short_term_capital_gains"),
            (
                "person",
                "puf_tax_itemization",
                "long_term_capital_gains_before_response",
            ),
            (
                "person",
                "puf_tax_itemization",
                "long_term_capital_gains_on_collectibles",
            ),
            ("person", "puf_tax_itemization", "non_sch_d_capital_gains"),
            (
                "person",
                "puf_tax_itemization",
                "taxable_private_pension_income",
            ),
            ("person", "puf_tax_itemization", "taxable_ira_distributions"),
            ("person", "puf_tax_itemization", "social_security_retirement"),
            ("person", "puf_tax_itemization", "social_security_disability"),
            ("person", "puf_tax_itemization", "social_security_dependents"),
            ("person", "puf_tax_itemization", "social_security_survivors"),
            ("person", "puf_tax_itemization", "alimony_income"),
            ("person", "puf_tax_itemization", "alimony_expense"),
            ("person", "puf_tax_itemization", "salt_refund_income"),
            ("person", "puf_tax_itemization", "charitable_cash_donations"),
            (
                "person",
                "puf_tax_itemization",
                "charitable_non_cash_donations",
            ),
            ("person", "puf_tax_itemization", "home_mortgage_interest"),
            (
                "person",
                "puf_tax_itemization",
                "investment_interest_expense",
            ),
            (
                "person",
                "puf_tax_itemization",
                "investment_income_elected_form_4952",
            ),
            ("person", "puf_tax_itemization", "student_loan_interest"),
            ("person", "puf_tax_itemization", "educator_expense"),
            ("person", "puf_tax_itemization", "qualified_tuition_expenses"),
            ("person", "puf_tax_itemization", "casualty_loss"),
            (
                "person",
                "puf_tax_itemization",
                "unreimbursed_business_employee_expenses",
            ),
            (
                "person",
                "puf_tax_itemization",
                "traditional_ira_contributions_desired",
            ),
            (
                "person",
                "puf_tax_itemization",
                "self_employed_pension_contributions_desired",
            ),
            ("person", "puf_tax_itemization", "rental_income"),
            ("person", "puf_tax_itemization", "estate_income"),
            ("person", "puf_tax_itemization", "farm_income"),
            ("person", "puf_tax_itemization", "farm_operations_income"),
            ("person", "puf_tax_itemization", "farm_rent_income"),
            ("person", "puf_tax_itemization", "miscellaneous_income"),
            ("person", "puf_tax_itemization", "partnership_income"),
            (
                "person",
                "puf_tax_itemization",
                "partnership_self_employment_net_earnings",
            ),
            ("person", "puf_tax_itemization", "qualified_bdc_income"),
            (
                "person",
                "puf_tax_itemization",
                "qualified_reit_and_ptp_income",
            ),
            (
                "person",
                "puf_tax_itemization",
                "sstb_self_employment_income_before_lsr",
            ),
            (
                "person",
                "puf_tax_itemization",
                "sstb_unadjusted_basis_qualified_property",
            ),
            (
                "person",
                "puf_tax_itemization",
                "sstb_w2_wages_from_qualified_business",
            ),
            (
                "person",
                "puf_tax_itemization",
                "unadjusted_basis_qualified_property",
            ),
            (
                "person",
                "puf_tax_itemization",
                "w2_wages_from_qualified_business",
            ),
            ("person", "simulated_output", "ssi"),
            (
                "person",
                "source_operator_child_support",
                "child_support_expense",
            ),
            (
                "person",
                "source_operator_child_support",
                "child_support_received",
            ),
            (
                "person",
                "source_operator_cps_carried",
                "strike_benefits",
            ),
            (
                "person",
                "source_operator_disability_benefits",
                "disability_benefits",
            ),
            (
                "person",
                "source_operator_education_inputs",
                "educational_assistance",
            ),
            (
                "person",
                "source_operator_hours_worked",
                "weekly_hours_worked_before_lsr",
            ),
            (
                "person",
                "source_operator_prior_year_income",
                "self_employment_income_last_year",
            ),
            (
                "person",
                "source_operator_retirement_contributions",
                "roth_401k_contributions_desired",
            ),
            (
                "person",
                "source_operator_retirement_contributions",
                "roth_ira_contributions_desired",
            ),
            (
                "person",
                "source_operator_retirement_contributions",
                "traditional_401k_contributions_desired",
            ),
            (
                "person",
                "source_operator_retirement_distributions",
                "keogh_distributions",
            ),
            (
                "person",
                "source_operator_retirement_distributions",
                "tax_exempt_ira_distributions",
            ),
            (
                "person",
                "source_operator_retirement_distributions",
                "taxable_401k_distributions",
            ),
            (
                "person",
                "source_operator_retirement_distributions",
                "taxable_403b_distributions",
            ),
            (
                "person",
                "source_operator_retirement_distributions",
                "taxable_sep_distributions",
            ),
            (
                "person",
                "source_operator_weeks_unemployed",
                "weeks_unemployed",
            ),
            (
                "person",
                "source_operator_workers_compensation",
                "workers_compensation",
            ),
            (
                "spm_unit",
                "model_required_numeric",
                "spm_unit_pre_subsidy_childcare_expenses",
            ),
            (
                "spm_unit",
                "source_operator_energy_subsidy",
                "spm_unit_energy_subsidy",
            ),
            ("tax_unit", "puf_tax_itemization", "domestic_production_ald"),
            (
                "tax_unit",
                "puf_tax_itemization",
                "unrecaptured_section_1250_gain",
            ),
            (
                "tax_unit",
                "puf_tax_itemization",
                "first_home_mortgage_balance",
            ),
            (
                "tax_unit",
                "puf_tax_itemization",
                "first_home_mortgage_interest",
            ),
            (
                "tax_unit",
                "puf_tax_itemization",
                "health_savings_account_ald",
            ),
        ),
        "boolean_incidence": (
            ("person", "adult_care", "is_incapable_of_self_care"),
            (
                "person",
                "model_required_boolean",
                "has_champva_health_coverage_at_interview",
            ),
            ("person", "model_required_boolean", "has_esi"),
            (
                "person",
                "model_required_boolean",
                "has_indian_health_service_coverage_at_interview",
            ),
            (
                "person",
                "model_required_boolean",
                "has_marketplace_health_coverage_at_interview",
            ),
            (
                "person",
                "model_required_boolean",
                "has_medicaid_health_coverage_at_interview",
            ),
            (
                "person",
                "model_required_boolean",
                "has_non_marketplace_direct_purchase_health_coverage_at_interview",
            ),
            (
                "person",
                "model_required_boolean",
                "has_other_means_tested_health_coverage_at_interview",
            ),
            (
                "person",
                "model_required_boolean",
                "has_tricare_health_coverage_at_interview",
            ),
            (
                "person",
                "model_required_boolean",
                "has_va_health_coverage_at_interview",
            ),
            ("person", "model_required_boolean", "is_blind"),
            ("person", "model_required_boolean", "is_disabled"),
            (
                "person",
                "model_required_boolean",
                "is_full_time_college_student",
            ),
            ("person", "model_required_boolean", "is_pregnant"),
            ("person", "model_required_boolean", "receives_wic"),
            (
                "person",
                "puf_tax_itemization",
                "estate_income_would_be_qualified",
            ),
            (
                "person",
                "puf_tax_itemization",
                "farm_operations_income_would_be_qualified",
            ),
            (
                "person",
                "puf_tax_itemization",
                "farm_rent_income_would_be_qualified",
            ),
            (
                "person",
                "puf_tax_itemization",
                "partnership_s_corp_income_would_be_qualified",
            ),
            (
                "person",
                "puf_tax_itemization",
                "rental_income_would_be_qualified",
            ),
            (
                "person",
                "puf_tax_itemization",
                "self_employment_income_would_be_qualified",
            ),
            (
                "person",
                "puf_tax_itemization",
                "sstb_self_employment_income_would_be_qualified",
            ),
            ("person", "puf_tax_itemization", "business_is_sstb"),
            (
                "person",
                "source_operator_education_inputs",
                "attends_eligible_educational_institution_for_american_opportunity_credit",
            ),
            (
                "person",
                "source_operator_education_inputs",
                "has_american_opportunity_credit_1098_t_or_exception",
            ),
            (
                "person",
                "source_operator_education_inputs",
                "has_american_opportunity_credit_institution_ein",
            ),
            (
                "person",
                "source_operator_education_inputs",
                "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            ),
            (
                "person",
                "source_operator_education_inputs",
                "is_pursuing_credential_for_american_opportunity_credit",
            ),
            (
                "person",
                "source_operator_medicare_take_up",
                "takes_up_medicare_if_eligible",
            ),
            (
                "person",
                "source_operator_prior_year_income",
                "previous_year_income_available",
            ),
            (
                "person",
                "source_operator_relationship_inputs",
                "is_separated",
            ),
            (
                "person",
                "source_operator_relationship_inputs",
                "is_surviving_spouse",
            ),
            (
                "person",
                "source_operator_wic_claim",
                "would_claim_wic",
            ),
            ("person", "take_up", "takes_up_basic_health_program_if_eligible"),
            ("person", "take_up", "takes_up_chip_if_eligible"),
            ("person", "take_up", "takes_up_early_head_start_if_eligible"),
            ("person", "take_up", "takes_up_head_start_if_eligible"),
            ("person", "take_up", "takes_up_medicaid_if_eligible"),
            ("person", "take_up", "takes_up_ssi_if_eligible"),
            (
                "spm_unit",
                "benefit_participation",
                "takes_up_housing_assistance_if_eligible",
            ),
            ("spm_unit", "model_required_boolean", "is_tanf_enrolled"),
            ("spm_unit", "model_required_boolean", "receives_snap"),
            (
                "spm_unit",
                "source_operator_housing_inputs",
                "receives_housing_assistance",
            ),
            ("spm_unit", "take_up", "takes_up_snap_if_eligible"),
            ("spm_unit", "take_up", "takes_up_tanf_if_eligible"),
            ("tax_unit", "take_up", "takes_up_aca_if_eligible"),
            ("tax_unit", "take_up", "takes_up_dc_ptc"),
            ("tax_unit", "take_up", "takes_up_eitc"),
        ),
        "categorical_tvd": (
            ("person", "model_required_discrete", "own_children_in_household"),
            (
                "person",
                "source_operator_immigration",
                "immigration_status_str",
            ),
            ("person", "source_operator_immigration", "ssn_card_type"),
            (
                "tax_unit",
                "puf_tax_itemization",
                "first_home_mortgage_origination_year",
            ),
        ),
    }
)


def _explicit_origin_battery_metric_registry(
    surface: TargetFamilies,
) -> Mapping[tuple[str, str, str, int], str]:
    registry = {
        (entity, family, column, 0): metric
        for metric, declarations in _EXPLICIT_ORIGIN_BATTERY_METRIC_DECLARATIONS.items()
        for entity, family, column in declarations
    }
    surface_keys = set(_surface_target_keys(surface))
    missing = sorted(surface_keys - set(registry))
    extra = sorted(set(registry) - surface_keys)
    if missing or extra:
        raise RuntimeError(
            "Explicit stacked battery metrics do not exactly cover the production "
            f"surface; missing={missing}, extra={extra}."
        )
    return _freeze_metric_registry(registry)


CANONICAL_STACKED_GAP_FILL_SURFACE = _freeze_target_families(
    pool_pre_clone_gap_fill_target_families()
)
CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE = _freeze_target_families(
    pool_post_puf_transfer_target_families()
)
CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE = _freeze_target_families(
    pool_post_puf_puf_producer_target_families()
)
CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE = _freeze_target_families(
    pool_post_puf_source_producer_target_families()
)
CANONICAL_STACKED_DECLARED_SURFACE = _terminal_surface_from_pool_registry()
CANONICAL_STACKED_GAP_FILL_PLAN = _build_stacked_gap_fill_plan(
    CANONICAL_STACKED_GAP_FILL_SURFACE
)
_CANONICAL_STACKED_GAP_FILL_PRODUCER_SCHEDULE = _build_gap_fill_producer_schedule(
    CANONICAL_STACKED_GAP_FILL_SURFACE
)
_canonical_producer_precedence_failures = _gap_fill_producer_precedence_failures(
    CANONICAL_STACKED_GAP_FILL_PLAN,
    _CANONICAL_STACKED_GAP_FILL_PRODUCER_SCHEDULE,
)
if _canonical_producer_precedence_failures:
    raise RuntimeError(
        "Canonical stacked gap-fill producer precedence is invalid:\n  "
        + "\n  ".join(_canonical_producer_precedence_failures)
    )
CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY = _explicit_origin_battery_metric_registry(
    CANONICAL_STACKED_DECLARED_SURFACE
)
CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY: Mapping[
    tuple[str, str, tuple[str, ...], int], str
] = MappingProxyType(
    {
        (
            "person",
            "source_operator_immigration",
            ("ssn_card_type", "immigration_status_str"),
            0,
        ): "categorical_tvd"
    }
)
CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE = _BatterySupportProfile(
    profile_id="us_stacked_origin_battery_support",
    version=1,
    min_effective_support=5,
)

_CANONICAL_STACKED_DECLARED_SURFACE_ANCHOR = CANONICAL_STACKED_DECLARED_SURFACE
_CANONICAL_STACKED_GAP_FILL_SURFACE_ANCHOR = CANONICAL_STACKED_GAP_FILL_SURFACE
_CANONICAL_STACKED_GAP_FILL_PLAN_ANCHOR = CANONICAL_STACKED_GAP_FILL_PLAN
_CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE_ANCHOR = (
    CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
)
_CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE_ANCHOR = (
    CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE
)
_CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE_ANCHOR = (
    CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE
)
_CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY_ANCHOR = (
    CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
)
_CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY_ANCHOR = (
    CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY
)
_CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE_ANCHOR = (
    CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE
)

# Active module-level authority references are intentionally separate from the
# immutable anchors. Rebinding any active reference is detected at evaluation,
# and the live content digest is receipted rather than trusting a stale hash.
_STACKED_DECLARED_SURFACE = CANONICAL_STACKED_DECLARED_SURFACE
_STACKED_GAP_FILL_SURFACE = CANONICAL_STACKED_GAP_FILL_SURFACE
_STACKED_GAP_FILL_PLAN = CANONICAL_STACKED_GAP_FILL_PLAN
_STACKED_POST_PUF_TRANSFER_SURFACE = CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
_STACKED_POST_PUF_PUF_PRODUCER_SURFACE = CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE
_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE = (
    CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE
)
_BATTERY_METRIC_REGISTRY = CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
_BATTERY_JOINT_METRIC_REGISTRY = CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY
_BATTERY_SUPPORT_PROFILE = CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE

_CANONICAL_STACKED_AUTHORITY = _make_stacked_authority(
    authority_id=_STACKED_AUTHORITY_ID,
    version=_STACKED_AUTHORITY_VERSION,
    gap_fill_plan=_CANONICAL_STACKED_GAP_FILL_PLAN_ANCHOR,
    post_puf_transfer_surface=(_CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE_ANCHOR),
    post_puf_puf_producer_surface=(
        _CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE_ANCHOR
    ),
    post_puf_source_producer_surface=(
        _CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE_ANCHOR
    ),
    declared_surface=_CANONICAL_STACKED_DECLARED_SURFACE_ANCHOR,
    metric_registry=_CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY_ANCHOR,
    joint_metric_registry=_CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY_ANCHOR,
    support_profile=_CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE_ANCHOR,
    declared_form=_CANONICAL_AUTHORITY_FORM,
)
_CANONICAL_STACKED_AUTHORITY_ANCHOR = _CANONICAL_STACKED_AUTHORITY


def _production_stacked_authority(
    *,
    _canonical_authority: _StackedAuthority = _CANONICAL_STACKED_AUTHORITY,
    _canonical_plan: tuple[GapFillDirection, ...] = CANONICAL_STACKED_GAP_FILL_PLAN,
    _canonical_gap_surface: TargetFamilies = CANONICAL_STACKED_GAP_FILL_SURFACE,
    _canonical_post_puf_surface: TargetFamilies = (
        CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
    ),
    _canonical_post_puf_puf_producer_surface: TargetFamilies = (
        CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE
    ),
    _canonical_post_puf_source_producer_surface: TargetFamilies = (
        CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE
    ),
    _canonical_surface: TargetFamilies = CANONICAL_STACKED_DECLARED_SURFACE,
    _canonical_registry: Mapping[
        tuple[str, str, str, int], str
    ] = CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY,
    _canonical_joint_registry: Mapping[
        tuple[str, str, tuple[str, ...], int], str
    ] = CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY,
    _canonical_profile: _BatterySupportProfile = (
        CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE
    ),
) -> _StackedAuthority:
    live_late_producer_schedule = us_late_producer_schedule_receipt()
    identity = (
        _STACKED_GAP_FILL_PLAN is _canonical_plan
        and _STACKED_GAP_FILL_SURFACE is _canonical_gap_surface
        and _STACKED_POST_PUF_TRANSFER_SURFACE is _canonical_post_puf_surface
        and _STACKED_POST_PUF_PUF_PRODUCER_SURFACE
        is _canonical_post_puf_puf_producer_surface
        and _STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE
        is _canonical_post_puf_source_producer_surface
        and _STACKED_DECLARED_SURFACE is _canonical_surface
        and _BATTERY_METRIC_REGISTRY is _canonical_registry
        and _BATTERY_JOINT_METRIC_REGISTRY is _canonical_joint_registry
        and _BATTERY_SUPPORT_PROFILE is _canonical_profile
        and _json_ready(live_late_producer_schedule)
        == _json_ready(_canonical_authority.late_producer_schedule)
    )
    if identity:
        return _canonical_authority
    return _make_stacked_authority(
        authority_id=_STACKED_AUTHORITY_ID,
        version=_STACKED_AUTHORITY_VERSION,
        gap_fill_plan=_STACKED_GAP_FILL_PLAN,
        post_puf_transfer_surface=_STACKED_POST_PUF_TRANSFER_SURFACE,
        post_puf_puf_producer_surface=_STACKED_POST_PUF_PUF_PRODUCER_SURFACE,
        post_puf_source_producer_surface=(_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE),
        declared_surface=_STACKED_DECLARED_SURFACE,
        metric_registry=_BATTERY_METRIC_REGISTRY,
        joint_metric_registry=_BATTERY_JOINT_METRIC_REGISTRY,
        support_profile=_BATTERY_SUPPORT_PROFILE,
        late_producer_schedule=live_late_producer_schedule,
        declared_form=_CANONICAL_AUTHORITY_FORM,
        declared_component_sha256=_canonical_authority.declared_component_sha256,
        declared_sha256=_canonical_authority.declared_sha256,
    )


def _metric_registry_for_surface(
    surface: TargetFamilies,
) -> Mapping[tuple[str, str, str, int], str]:
    inferred = _test_metric_registry_for_surface(surface)
    return MappingProxyType(
        {
            key: CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY.get(
                key,
                inferred[key],
            )
            for key in _surface_target_keys(surface)
        }
    )


def _restrict_surface_to_declared_targets(
    surface: TargetFamilies,
    declared: TargetFamilies,
) -> TargetFamilies:
    declared_keys = set(_surface_target_keys(declared))
    restricted: dict[str, dict[str, tuple[str, ...]]] = {}
    for entity, families in surface.items():
        for family, targets in families.items():
            retained = tuple(
                target
                for target in targets
                if (entity, family, target, 0) in declared_keys
            )
            if retained:
                restricted.setdefault(entity, {})[family] = retained
    return restricted


def _make_test_stacked_authority(
    *,
    declared_surface: TargetFamilies | None = None,
    gap_fill_plan: Sequence[GapFillDirection] | None = None,
    post_puf_transfer_surface: TargetFamilies | None = None,
    post_puf_puf_producer_surface: TargetFamilies | None = None,
    post_puf_source_producer_surface: TargetFamilies | None = None,
    metric_registry: Mapping[tuple[str, str, str, int], str] | None = None,
    joint_metric_registry: Mapping[tuple[str, str, tuple[str, ...], int], str]
    | None = None,
    support_profile: _BatterySupportProfile | None = None,
) -> _StackedAuthority:
    """Explicit test-only seam; every receipt is marked non-canonical."""

    surface = (
        CANONICAL_STACKED_DECLARED_SURFACE
        if declared_surface is None
        else declared_surface
    )
    plan = CANONICAL_STACKED_GAP_FILL_PLAN if gap_fill_plan is None else gap_fill_plan
    post_puf_surface = (
        {} if post_puf_transfer_surface is None else post_puf_transfer_surface
    )
    puf_producer_surface = (
        _restrict_surface_to_declared_targets(
            CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE,
            post_puf_surface,
        )
        if post_puf_puf_producer_surface is None
        else post_puf_puf_producer_surface
    )
    source_producer_surface = (
        _restrict_surface_to_declared_targets(
            CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE,
            post_puf_surface,
        )
        if post_puf_source_producer_surface is None
        else post_puf_source_producer_surface
    )
    registry = (
        _metric_registry_for_surface(surface)
        if metric_registry is None
        else metric_registry
    )
    surface_keys = set(_surface_target_keys(surface))
    joints = (
        {
            key: metric
            for key, metric in CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY.items()
            if all(
                (key[0], key[1], column, key[3]) in surface_keys for column in key[2]
            )
        }
        if joint_metric_registry is None
        else joint_metric_registry
    )
    return _make_stacked_authority(
        authority_id=f"{_STACKED_AUTHORITY_ID}.test",
        version=_STACKED_AUTHORITY_VERSION,
        gap_fill_plan=plan,
        post_puf_transfer_surface=post_puf_surface,
        post_puf_puf_producer_surface=puf_producer_surface,
        post_puf_source_producer_surface=source_producer_surface,
        declared_surface=surface,
        metric_registry=registry,
        joint_metric_registry=joints,
        support_profile=(
            CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE
            if support_profile is None
            else support_profile
        ),
        declared_form=_NONCANONICAL_AUTHORITY_FORM,
    )


def stacked_gap_fill_plan() -> tuple[GapFillDirection, ...]:
    """Return the immutable canonical two-direction stacked gap-fill plan."""

    return _STACKED_GAP_FILL_PLAN


def stacked_gap_fill_producer_schedule_receipt() -> Mapping[str, object]:
    """Return the live channel-aware proof that every producer precedes its check."""

    plan = stacked_gap_fill_plan()
    schedule = _build_gap_fill_producer_schedule(
        pool_pre_clone_gap_fill_target_families()
    )
    failures = _gap_fill_producer_precedence_failures(plan, schedule)
    if failures:
        raise ValueError(
            "Stacked gap-fill producer precedence failed:\n  " + "\n  ".join(failures)
        )
    schedule_by_target = {
        (record.entity, record.family, record.target): record for record in schedule
    }
    directions: list[dict[str, object]] = []
    for index, direction in enumerate(plan):
        targets: list[dict[str, object]] = []
        for entity, families in direction.target_families.items():
            for family, columns in families.items():
                for column in columns:
                    record = schedule_by_target[(entity, family, column)]
                    targets.append(
                        {
                            "entity": entity,
                            "family": family,
                            "column": column,
                            "producer": record.operator,
                            "producer_order_index": record.operator_order_index,
                            "execution_scope": record.execution_scope,
                            "produced_channel": record.produced_channel,
                            "producer_stage": record.producer_stage,
                        }
                    )
        directions.append(
            {
                "name": direction.name,
                "order_index": index,
                "donor_channel": direction.donor_channel,
                "activation_stage": _gap_fill_activation_stage(direction.name),
                "target_count": len(targets),
                "targets": targets,
            }
        )
    payload: dict[str, object] = {
        "status": "all_producers_precede_activation",
        "direction_count": len(directions),
        "target_count": len(schedule),
        "directions": directions,
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def stacked_spine_authority_receipt() -> Mapping[str, object]:
    """Return the live-digested canonical authority for build identity binding."""

    authority = _production_stacked_authority()
    receipt = _authority_receipt(authority)
    _validate_production_authority_receipt(
        receipt,
        boundary="stacked spine authority identity",
    )
    return _json_ready(receipt)


def _direction_target_index(
    plan: Sequence[GapFillDirection],
) -> dict[tuple[str, str, str, int], GapFillDirection]:
    index: dict[tuple[str, str, str, int], GapFillDirection] = {}
    for direction in plan:
        for entity, families in direction.target_families.items():
            for family, targets in families.items():
                for target in targets:
                    index[(entity, family, target, 0)] = direction
    return index


def _direction_signature(direction: GapFillDirection) -> tuple[str, str, str]:
    return (
        direction.name,
        direction.recipient_channel,
        direction.donor_channel,
    )


def _authority_receipt(
    authority: _StackedAuthority,
    *,
    _canonical_authority: _StackedAuthority = _CANONICAL_STACKED_AUTHORITY,
) -> dict[str, object]:
    """Receipt live content, claimed digests, identity, and component counts."""

    live_components, live_bundle = _authority_live_digests(authority)
    component_integrity = {
        name: digest == authority.declared_component_sha256[name]
        for name, digest in live_components.items()
    }
    integrity = all(component_integrity.values()) and (
        live_bundle == authority.declared_sha256
    )
    canonical_identity = authority is _canonical_authority
    canonical_content = (
        authority.authority_id == _STACKED_AUTHORITY_ID
        and authority.version == _STACKED_AUTHORITY_VERSION
        and live_components == dict(_canonical_authority.declared_component_sha256)
        and live_bundle == _canonical_authority.declared_sha256
    )
    canonical = (
        authority.declared_form == _CANONICAL_AUTHORITY_FORM
        and canonical_identity
        and canonical_content
        and integrity
    )
    support = _support_profile_payload(authority.support_profile)
    components: dict[str, dict[str, object]] = {
        "gap_fill_plan": {
            "sha256": live_components["gap_fill_plan"],
            "declared_sha256": authority.declared_component_sha256["gap_fill_plan"],
            "target_count": len(_plan_target_keys(authority.gap_fill_plan)),
            "direction_count": len(authority.gap_fill_plan),
            "digest_matches_declared": component_integrity["gap_fill_plan"],
        },
        "post_puf_transfer_surface": {
            "sha256": live_components["post_puf_transfer_surface"],
            "declared_sha256": authority.declared_component_sha256[
                "post_puf_transfer_surface"
            ],
            "target_count": len(
                _surface_target_keys(authority.post_puf_transfer_surface)
            ),
            "puf_producer_target_count": len(
                _surface_target_keys(authority.post_puf_puf_producer_surface)
            ),
            "source_producer_target_count": len(
                _surface_target_keys(authority.post_puf_source_producer_surface)
            ),
            "donor_channel": BASE_ASEC_SUPPORT_CHANNEL,
            "donor_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
            "recipient_selection": (
                "target_specific_complement_of_declared_producer_rows"
            ),
            "digest_matches_declared": component_integrity["post_puf_transfer_surface"],
        },
        "declared_surface": {
            "sha256": live_components["declared_surface"],
            "declared_sha256": authority.declared_component_sha256["declared_surface"],
            "target_count": len(_surface_target_keys(authority.declared_surface)),
            "entity_count": len(authority.declared_surface),
            "digest_matches_declared": component_integrity["declared_surface"],
        },
        "metric_registry": {
            "sha256": live_components["metric_registry"],
            "declared_sha256": authority.declared_component_sha256["metric_registry"],
            "target_count": len(authority.metric_registry),
            "digest_matches_declared": component_integrity["metric_registry"],
        },
        "joint_metric_registry": {
            "sha256": live_components["joint_metric_registry"],
            "declared_sha256": authority.declared_component_sha256[
                "joint_metric_registry"
            ],
            "target_count": len(authority.joint_metric_registry),
            "digest_matches_declared": component_integrity["joint_metric_registry"],
        },
        "support_profile": {
            **support,
            "sha256": live_components["support_profile"],
            "declared_sha256": authority.declared_component_sha256["support_profile"],
            "digest_matches_declared": component_integrity["support_profile"],
        },
        "puf_capital_gains_tail_support_contract": {
            "identity": _json_ready(authority.puf_capital_gains_tail_support_contract),
            "sha256": live_components["puf_capital_gains_tail_support_contract"],
            "declared_sha256": authority.declared_component_sha256[
                "puf_capital_gains_tail_support_contract"
            ],
            "digest_matches_declared": component_integrity[
                "puf_capital_gains_tail_support_contract"
            ],
        },
        "late_producer_schedule": {
            "identity": _json_ready(authority.late_producer_schedule),
            "sha256": live_components["late_producer_schedule"],
            "declared_sha256": authority.declared_component_sha256[
                "late_producer_schedule"
            ],
            "schedule_sha256": authority.late_producer_schedule.get("schedule_sha256"),
            "producer_count": authority.late_producer_schedule.get("producer_count"),
            "digest_matches_declared": component_integrity["late_producer_schedule"],
        },
    }
    return {
        "authority_id": authority.authority_id,
        "version": authority.version,
        "authority_form": (
            _CANONICAL_AUTHORITY_FORM if canonical else _NONCANONICAL_AUTHORITY_FORM
        ),
        "declared_authority_form": authority.declared_form,
        "canonical": canonical,
        "production_manifest_permitted": canonical,
        "canonical_identity": canonical_identity,
        "canonical_content": canonical_content,
        "integrity_valid": integrity,
        "sha256": live_bundle,
        "declared_sha256": authority.declared_sha256,
        "digest_matches_declared": live_bundle == authority.declared_sha256,
        "components": components,
    }


def _authority_validation_failures(
    authority: _StackedAuthority,
    *,
    production: bool,
    _canonical_plan: tuple[GapFillDirection, ...] = CANONICAL_STACKED_GAP_FILL_PLAN,
    _canonical_registry: Mapping[
        tuple[str, str, str, int], str
    ] = CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY,
    _canonical_joint_registry: Mapping[
        tuple[str, str, tuple[str, ...], int], str
    ] = CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY,
    _canonical_profile: _BatterySupportProfile = (
        CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE
    ),
) -> list[str]:
    receipt = _authority_receipt(authority)
    failures: list[str] = []
    surface_targets = _surface_target_keys(authority.declared_surface)
    plan_targets = _plan_target_keys(authority.gap_fill_plan)
    post_puf_targets = _surface_target_keys(authority.post_puf_transfer_surface)
    post_puf_puf_producer_targets = _surface_target_keys(
        authority.post_puf_puf_producer_surface
    )
    post_puf_source_producer_targets = _surface_target_keys(
        authority.post_puf_source_producer_surface
    )
    duplicate_surface_targets = sorted(
        target for target, count in Counter(surface_targets).items() if count > 1
    )
    duplicate_plan_targets = sorted(
        target for target, count in Counter(plan_targets).items() if count > 1
    )
    duplicate_post_puf_targets = sorted(
        target for target, count in Counter(post_puf_targets).items() if count > 1
    )
    duplicate_post_puf_producer_targets = sorted(
        {
            target
            for targets in (
                post_puf_puf_producer_targets,
                post_puf_source_producer_targets,
            )
            for target, count in Counter(targets).items()
            if count > 1
        }
    )
    if production:
        failures.extend(
            _gap_fill_producer_precedence_failures(
                authority.gap_fill_plan,
                _build_gap_fill_producer_schedule(
                    pool_pre_clone_gap_fill_target_families()
                ),
            )
        )
    if duplicate_surface_targets:
        failures.append(
            "declared surface repeats target(s): "
            + ", ".join(
                _battery_target_label(target) for target in duplicate_surface_targets
            )
            + "."
        )
    if duplicate_plan_targets:
        failures.append(
            "gap-fill plan repeats target(s): "
            + ", ".join(
                _battery_target_label(target) for target in duplicate_plan_targets
            )
            + "."
        )
    if duplicate_post_puf_targets:
        failures.append(
            "post-PUF transfer surface repeats target(s): "
            + ", ".join(
                _battery_target_label(target) for target in duplicate_post_puf_targets
            )
            + "."
        )
    if duplicate_post_puf_producer_targets:
        failures.append(
            "post-PUF producer surfaces repeat target(s) within a role: "
            + ", ".join(
                _battery_target_label(target)
                for target in duplicate_post_puf_producer_targets
            )
            + "."
        )
    overlap = sorted(set(plan_targets) & set(post_puf_targets))
    if overlap:
        failures.append(
            "early gap-fill and post-PUF transfer surfaces overlap: "
            + ", ".join(_battery_target_label(target) for target in overlap)
            + "."
        )
    outside_declared = sorted(set(post_puf_targets) - set(surface_targets))
    if outside_declared:
        failures.append(
            "post-PUF transfer targets are absent from the declared terminal "
            "surface: "
            + ", ".join(_battery_target_label(target) for target in outside_declared)
            + "."
        )
    producer_targets = set(post_puf_puf_producer_targets) | set(
        post_puf_source_producer_targets
    )
    unowned_post_puf = sorted(set(post_puf_targets) - producer_targets)
    outside_post_puf = sorted(producer_targets - set(post_puf_targets))
    if unowned_post_puf:
        failures.append(
            "post-PUF transfer targets have no declared producer role: "
            + ", ".join(_battery_target_label(target) for target in unowned_post_puf)
            + "."
        )
    if outside_post_puf:
        failures.append(
            "post-PUF producer roles name targets outside the transfer surface: "
            + ", ".join(_battery_target_label(target) for target in outside_post_puf)
            + "."
        )
    for name, label in (
        ("gap_fill_plan", "gap-fill plan"),
        ("post_puf_transfer_surface", "post-PUF transfer surface"),
        ("declared_surface", "declared surface"),
        ("metric_registry", "metric registry"),
        ("joint_metric_registry", "joint metric registry"),
        ("support_profile", "support profile"),
        (
            "puf_capital_gains_tail_support_contract",
            "PUF capital-gains-tail support contract",
        ),
        ("late_producer_schedule", "late producer schedule"),
    ):
        component = receipt["components"][name]
        if not component["digest_matches_declared"]:
            failures.append(
                f"{label} live-content digest mismatch: declared "
                f"{component['declared_sha256']}, computed {component['sha256']}."
            )
    if not receipt["digest_matches_declared"]:
        failures.append(
            "stacked authority live-content digest mismatch: declared "
            f"{receipt['declared_sha256']}, computed {receipt['sha256']}."
        )

    canonical_directions = _direction_target_index(_canonical_plan)
    for target, direction in _direction_target_index(authority.gap_fill_plan).items():
        canonical_direction = canonical_directions.get(target)
        if canonical_direction is not None and _direction_signature(
            direction
        ) != _direction_signature(canonical_direction):
            failures.append(
                f"canonical gap-fill direction mismatch for "
                f"{_battery_target_label(target)}: authoritative "
                f"{_direction_signature(canonical_direction)!r}, got "
                f"{_direction_signature(direction)!r}."
            )
    for target, metric in authority.metric_registry.items():
        canonical_metric = _canonical_registry.get(target)
        if canonical_metric is not None and metric != canonical_metric:
            failures.append(
                f"declared battery target {_battery_target_label(target)} must "
                f"use authoritative metric {canonical_metric!r}, got {metric!r}."
            )
    for target, metric in authority.joint_metric_registry.items():
        entity, family, columns, clone_index = target
        missing_members = [
            column
            for column in columns
            if (entity, family, column, clone_index) not in authority.metric_registry
        ]
        noncategorical_members = [
            column
            for column in columns
            if authority.metric_registry.get((entity, family, column, clone_index))
            not in {None, "categorical_tvd"}
        ]
        if missing_members:
            failures.append(
                f"joint battery target {_joint_battery_target_label(target)} has "
                f"unregistered member(s) {missing_members}."
            )
        if noncategorical_members:
            failures.append(
                f"joint battery target {_joint_battery_target_label(target)} has "
                f"non-categorical member metric(s) {noncategorical_members}."
            )
        canonical_metric = _canonical_joint_registry.get(target)
        if canonical_metric is not None and metric != canonical_metric:
            failures.append(
                f"declared joint battery target {_joint_battery_target_label(target)} "
                f"must use authoritative metric {canonical_metric!r}, got "
                f"{metric!r}."
            )
    if authority.support_profile != _canonical_profile:
        failures.append(
            "support profile differs from the canonical stacked battery profile."
        )
    if production and receipt["canonical_identity"] is not True:
        failures.append("canonical stacked authority identity mismatch.")
    if production and receipt["canonical_content"] is not True:
        failures.append("canonical stacked authority live content mismatch.")
    if production and not receipt["canonical"]:
        failures.append(
            "non-canonical stacked authority is forbidden in production manifests."
        )
    return failures


def _validate_production_authority_receipt(
    receipt: Mapping[str, object],
    *,
    boundary: str,
    _canonical_authority: _StackedAuthority = _CANONICAL_STACKED_AUTHORITY,
) -> None:
    """Terminally reject any non-canonical authority at artifact emission."""

    expected = _authority_receipt(_canonical_authority)
    if dict(receipt) != expected:
        raise ValueError(
            f"{boundary}: non-canonical stacked authority is forbidden; "
            "production manifest emission is forbidden."
        )


def validate_stacked_post_puf_transfer_receipt(
    receipt: Mapping[str, object],
    *,
    boundary: str,
) -> None:
    """Reject a late-transfer receipt unless its full DAG proof is canonical."""

    if not isinstance(receipt, Mapping):
        raise ValueError(f"{boundary}: stacked post-PUF transfer receipt is absent.")
    authority = receipt.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer receipt has no authority; "
            "production manifest emission is forbidden."
        )
    _validate_production_authority_receipt(authority, boundary=boundary)
    schedule = receipt.get("producer_schedule")
    expected_schedule = _json_ready(us_late_producer_schedule_receipt())
    if not isinstance(schedule, Mapping) or _json_ready(schedule) != expected_schedule:
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer receipt has no canonical "
            "late-producer schedule; production manifest emission is forbidden."
        )
    expected_execution_order = [
        producer
        for producer in CANONICAL_US_LATE_PRODUCER_SCHEDULE.order
        if producer != US_LATE_PRIMARY_PUF_STAGE
    ]
    if receipt.get("producer_execution_order") != expected_execution_order:
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer execution order does not "
            "match the derived late-producer schedule; production manifest "
            "emission is forbidden."
        )
    expected_groups = {group.name: group for group in CANONICAL_US_LATE_TRANSFER_GROUPS}
    groups = receipt.get("groups")
    if not isinstance(groups, Mapping) or set(groups) != set(expected_groups):
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer group surface is not the "
            "canonical 19-group partition; production manifest emission is "
            "forbidden."
        )
    for name, group in expected_groups.items():
        group_receipt = groups[name]
        if (
            not isinstance(group_receipt, Mapping)
            or group_receipt.get("producer") != name
            or tuple(group_receipt.get("ordered_targets", ())) != group.targets
        ):
            raise ValueError(
                f"{boundary}: stacked post-PUF transfer group {name!r} is "
                "misbound; production manifest emission is forbidden."
            )
    expected_target_labels = {
        f"{entity}/{family}/{target}"
        for entity, families in CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        for family, targets in families.items()
        for target in targets
    }
    targets = receipt.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != expected_target_labels:
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer target surface is not the "
            "canonical 70-target surface; production manifest emission is "
            "forbidden."
        )
    if any(
        not isinstance(target_receipt, Mapping)
        or target_receipt.get("residual_null_rows") != 0
        for target_receipt in targets.values()
    ):
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer target receipts do not "
            "prove zero residual nulls; production manifest emission is forbidden."
        )
    completion = receipt.get("completion")
    if completion != {
        "status": "complete",
        "group_count": 19,
        "target_count": 70,
        "residual_null_rows": 0,
    }:
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer completion receipt is not "
            "canonical; production manifest emission is forbidden."
        )


_LATE_TABLE_DIGEST_CODEC = "canonical_scalar_v1"
_LATE_PRIMARY_QRF_INPUT_BINDING_ARTIFACT_KIND = (
    "populace_us_stacked_late_primary_qrf_input_binding"
)
_LATE_PRIMARY_QRF_INPUT_BINDING_FILENAME = "late-producer-input-binding.json"
_LATE_RESOURCE_SEMANTICS_ARTIFACT_KIND = (
    "populace_us_stacked_late_producer_resource_semantics"
)
_LATE_TABLE_DIGEST_CHUNK_ROWS = 65_536


def _late_digest_part(
    digest,
    *,
    domain: str,
    payload: bytes | bytearray | memoryview | np.ndarray,
) -> None:
    """Append one length-framed, domain-separated byte field to a digest."""

    domain_bytes = domain.encode("utf-8")
    payload_view = memoryview(payload)
    if payload_view.format != "B" or payload_view.ndim != 1:
        payload_view = payload_view.cast("B")
    digest.update(struct.pack("<I", len(domain_bytes)))
    digest.update(domain_bytes)
    digest.update(struct.pack("<Q", payload_view.nbytes))
    digest.update(payload_view)


def _late_little_endian_bytes(values: np.ndarray) -> np.ndarray:
    """Return a contiguous, explicitly little-endian numeric byte source."""

    array = np.asarray(values)
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def _late_scalar_bytes(value: object) -> bytes:
    """Encode one supported object scalar without lossy intermediary hashes."""

    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return b"null"
    if isinstance(value, (bool, np.bool_)):
        return b"bool\x01" if bool(value) else b"bool\x00"
    if isinstance(value, (int, np.integer)):
        return b"integer\x00" + str(int(value)).encode("ascii")
    if isinstance(value, (float, np.floating)):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            raise TypeError(
                "US late-producer content digest does not support object "
                f"floating scalar {value.dtype!s}."
            )
        return b"float64\x00" + struct.pack("<d", float(value))
    if isinstance(value, (complex, np.complexfloating)):
        if isinstance(value, np.complexfloating) and value.dtype.itemsize > 16:
            raise TypeError(
                "US late-producer content digest does not support object "
                f"complex scalar {value.dtype!s}."
            )
        numeric = complex(value)
        return b"complex128\x00" + struct.pack("<dd", numeric.real, numeric.imag)
    if isinstance(value, str):
        return b"string\x00" + value.encode("utf-8")
    if isinstance(value, (bytes, np.bytes_)):
        return b"bytes\x00" + bytes(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return b"datetime64ns\x00" + struct.pack("<q", timestamp.value)
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        delta = pd.Timedelta(value)
        return b"timedelta64ns\x00" + struct.pack("<q", delta.value)
    raise TypeError(
        "US late-producer content digest received unsupported object scalar "
        f"{type(value).__name__}."
    )


def _late_digest_variable_width_values(
    digest,
    values: Sequence[object],
    missing: np.ndarray,
    *,
    domain: str,
    strings_only: bool,
) -> None:
    """Stream framed string or object scalars in bounded-memory chunks."""

    for chunk_index, start in enumerate(
        range(0, len(values), _LATE_TABLE_DIGEST_CHUNK_ROWS)
    ):
        stop = min(start + _LATE_TABLE_DIGEST_CHUNK_ROWS, len(values))
        lengths = np.zeros(stop - start, dtype="<u8")
        payload = bytearray()
        for local_index, value in enumerate(values[start:stop]):
            if missing[start + local_index]:
                encoded = b""
            elif strings_only:
                if not isinstance(value, str):
                    raise TypeError(
                        "US late-producer string digest received non-string "
                        f"scalar {type(value).__name__}."
                    )
                encoded = value.encode("utf-8")
            else:
                encoded = _late_scalar_bytes(value)
            lengths[local_index] = len(encoded)
            payload.extend(encoded)
        chunk_domain = f"{domain}/chunk/{chunk_index}"
        _late_digest_part(
            digest,
            domain=f"{chunk_domain}/lengths",
            payload=lengths,
        )
        _late_digest_part(
            digest,
            domain=f"{chunk_domain}/payload",
            payload=payload,
        )


def _late_digest_series_values(
    digest,
    series: pd.Series,
    *,
    domain: str,
) -> None:
    """Hash one ordered logical Series with explicit dtype and null domains."""

    dtype = series.dtype
    missing = series.isna().to_numpy(dtype=bool)
    _late_digest_part(
        digest,
        domain=f"{domain}/dtype",
        payload=str(dtype).encode("utf-8"),
    )
    _late_digest_part(
        digest,
        domain=f"{domain}/row_count",
        payload=struct.pack("<Q", len(series)),
    )
    _late_digest_part(
        digest,
        domain=f"{domain}/null_bitmap",
        payload=np.ascontiguousarray(missing, dtype=np.uint8),
    )

    if pd.api.types.is_bool_dtype(dtype):
        encoded = series.to_numpy(dtype=np.uint8, na_value=0)
    elif pd.api.types.is_integer_dtype(dtype):
        numpy_dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
        encoded = series.to_numpy(dtype=numpy_dtype, na_value=0)
    elif pd.api.types.is_float_dtype(dtype):
        numpy_dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
        encoded = series.to_numpy(dtype=numpy_dtype, na_value=0.0)
        if missing.any():
            encoded = encoded.copy()
            encoded[missing] = 0.0
    elif pd.api.types.is_complex_dtype(dtype):
        numpy_dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
        encoded = series.to_numpy(dtype=numpy_dtype, na_value=0.0j)
        if missing.any():
            encoded = encoded.copy()
            encoded[missing] = 0.0j
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        encoded = series.array.asi8.copy()
        encoded[missing] = 0
    elif pd.api.types.is_timedelta64_dtype(dtype):
        encoded = series.array.asi8.copy()
        encoded[missing] = 0
    elif isinstance(dtype, pd.StringDtype):
        _late_digest_variable_width_values(
            digest,
            series.array,
            missing,
            domain=f"{domain}/strings",
            strings_only=True,
        )
        return
    else:
        _late_digest_variable_width_values(
            digest,
            series.array,
            missing,
            domain=f"{domain}/objects",
            strings_only=False,
        )
        return

    _late_digest_part(
        digest,
        domain=f"{domain}/fixed_width_values",
        payload=_late_little_endian_bytes(encoded),
    )


def _late_table_values_sha256(
    table: pd.DataFrame,
    *,
    normalize_strings: bool = False,
) -> str:
    """Hash ordered table scalars directly with typed, null-aware framing."""

    values = (
        canonicalize_table_string_dtypes(
            table,
            boundary="late-producer content digest",
            table_name="declared_surface",
        )
        if normalize_strings
        else table
    )
    if isinstance(values.index, pd.MultiIndex):
        index_levels = [
            pd.Series(values.index.get_level_values(level), copy=False)
            for level in range(values.index.nlevels)
        ]
    else:
        index_levels = [pd.Series(values.index, copy=False)]
    header = {
        "codec": _LATE_TABLE_DIGEST_CODEC,
        "columns": [str(column) for column in values.columns],
        "dtypes": [
            str(values.iloc[:, index].dtype) for index in range(values.shape[1])
        ],
        "index_type": type(values.index).__name__,
        "index_dtype": str(values.index.dtype),
        "index_level_dtypes": [str(level.dtype) for level in index_levels],
        "index_names": [
            None if name is None else str(name) for name in values.index.names
        ],
    }
    digest = hashlib.sha256()
    _late_digest_part(
        digest,
        domain="late_table_digest_codec",
        payload=_LATE_TABLE_DIGEST_CODEC.encode("ascii"),
    )
    _late_digest_part(
        digest,
        domain="late_table_header_json",
        payload=json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
    for level_index, level in enumerate(index_levels):
        _late_digest_series_values(
            digest,
            level,
            domain=f"index_level/{level_index}",
        )
    for column_index in range(values.shape[1]):
        _late_digest_series_values(
            digest,
            values.iloc[:, column_index],
            domain=f"column/{column_index}",
        )
    return digest.hexdigest()


def _late_virtual_resource_kind(column: str) -> str:
    """Return the exact semantic kind for one declared virtual input."""

    kinds = {
        "@puf_donor_tax_units": "puf_donor_tax_units",
        "@primary_qrf_checkpoint": "primary_qrf_checkpoint",
        US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT: (
            "acs_pums_earnings_universe_execution_config"
        ),
        US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT: "primary_puf_execution_config",
        US_LATE_SOURCE_EXECUTION_CONFIG_INPUT: ("post_clone_source_execution_config"),
        US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT: (
            "source_finalizer_execution_config"
        ),
        US_LATE_TRANSFER_MODEL_CONFIG_INPUT: "late_transfer_model_config",
        US_LATE_TRANSFER_TARGET_BANK_INPUT: "late_transfer_target_bank",
    }
    if column.startswith("@source_receipt:"):
        return "source_operator_receipt"
    try:
        return kinds[column]
    except KeyError as exc:
        raise ValueError(
            f"Unknown US late-producer virtual resource input {column!r}."
        ) from exc


def _late_resource_binding_schema_version(column: str) -> int:
    """Return the independently versioned payload schema for one resource."""

    kind = _late_virtual_resource_kind(column)
    return {
        "acs_pums_earnings_universe_execution_config": 2,
        "primary_puf_execution_config": 3,
        "post_clone_source_execution_config": 3,
        "source_finalizer_execution_config": 2,
        "late_transfer_model_config": 3,
    }.get(kind, 1)


def _late_contract_available_input_keys(
    contract: ProducerContract,
) -> set[str]:
    """Return the exact external-receipt keys required by one contract."""

    return {
        f"{column.entity}.{column.column}"
        for requirement in contract.inputs
        for alternative in requirement.alternatives
        for column in alternative
        if column.column.startswith("@")
        and column.column != "@resolved_weight"
        and column.entity != "frame"
    }


def _validate_late_resource_binding(
    binding: Mapping[str, object],
    *,
    producer: str,
    entity: str,
    column: str,
    boundary: str,
) -> None:
    """Reject hash-consistent resource claims with incomplete semantics."""

    kind = _late_virtual_resource_kind(column)

    def require_keys(expected: set[str]) -> None:
        if set(binding) != expected:
            raise ValueError(
                f"{boundary}: late resource {entity}.{column} {kind!r} binding "
                f"schema drifted; missing={sorted(expected - set(binding))}, "
                f"extra={sorted(set(binding) - expected)}."
            )

    def require_nonnegative_integer(value: object, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{boundary}: late resource {entity}.{column} has invalid "
                f"{label}={value!r}."
            )
        return value

    def require_positive_integer(value: object, *, label: str) -> int:
        result = require_nonnegative_integer(value, label=label)
        if result == 0:
            raise ValueError(
                f"{boundary}: late resource {entity}.{column} requires positive "
                f"{label}."
            )
        return result

    common = {"resource_kind", "schema_version"}
    if kind == "puf_donor_tax_units":
        require_keys({*common, "table_content_sha256", "ordered_columns", "dtypes"})
        _validate_sha256(
            binding.get("table_content_sha256"),
            boundary=f"{boundary} PUF donor content",
        )
        columns = binding.get("ordered_columns")
        dtypes = binding.get("dtypes")
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(value, str) or not value for value in columns)
            or not isinstance(dtypes, list)
            or len(dtypes) != len(columns)
            or any(not isinstance(value, str) or not value for value in dtypes)
        ):
            raise ValueError(
                f"{boundary}: late PUF donor binding has malformed columns/dtypes."
            )
        return
    if kind == "primary_qrf_checkpoint":
        require_keys(
            {
                *common,
                "checkpoint_identity_sha256",
                "checkpoint_schema_version",
                "manifest_filename",
                "mode",
                "target_order",
                "target_order_sha256",
            }
        )
        _validate_sha256(
            binding.get("checkpoint_identity_sha256"),
            boundary=f"{boundary} primary-QRF checkpoint identity",
        )
        expected = {
            "mode": "identity_bound_checkpoint",
            "checkpoint_schema_version": PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
            "manifest_filename": PRIMARY_QRF_MANIFEST_FILENAME,
            "target_order": list(PRIMARY_QRF_TARGET_ORDER),
            "target_order_sha256": PRIMARY_QRF_TARGET_ORDER_SHA256,
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"{boundary}: late primary-QRF checkpoint semantics changed."
            )
        return
    if kind == "acs_pums_earnings_universe_execution_config":
        require_keys(
            {
                *common,
                "runtime_identity_owner",
                "ordered_mapped_columns",
                "person_scope_mode",
                "contract_identity",
            }
        )
        if binding.get("ordered_mapped_columns") != list(
            ACS_PUMS_EARNINGS_SOURCE_COLUMNS
        ):
            raise ValueError(
                f"{boundary}: late ACS earnings-universe target order changed."
            )
        if binding.get("person_scope_mode") != "whole_frame_acs_channel":
            raise ValueError(
                f"{boundary}: late ACS earnings-universe scope mode changed."
            )
        if (
            binding.get("contract_identity")
            != _CANONICAL_ACS_EARNINGS_UNIVERSE_CONTRACT_IDENTITY
        ):
            raise ValueError(
                f"{boundary}: late ACS earnings-universe contract changed."
            )
        if binding.get("runtime_identity_owner") != (
            "microcosm.build.us_runtime.acs_income_universe"
        ):
            raise ValueError(
                f"{boundary}: late ACS earnings-universe runtime owner changed."
            )
        return
    if kind == "primary_puf_execution_config":
        require_keys(
            {
                *common,
                "clone_attachment",
                "qrf",
                "doctrines",
                "capital_gains_tail",
                "audit_sinks",
            }
        )
        clone = binding.get("clone_attachment")
        qrf = binding.get("qrf")
        doctrines = binding.get("doctrines")
        tail = binding.get("capital_gains_tail")
        audit_sinks = binding.get("audit_sinks")
        if not isinstance(clone, Mapping) or set(clone) != {
            "fraction",
            "seed",
            "support_channels",
            "puf_clone_index",
        }:
            raise ValueError(f"{boundary}: late clone-attachment config is malformed.")
        fraction = clone.get("fraction")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not np.isfinite(fraction)
            or not 0 < float(fraction) <= 1
        ):
            raise ValueError(f"{boundary}: late clone-attachment fraction is invalid.")
        require_nonnegative_integer(clone.get("seed"), label="clone seed")
        if (
            clone.get("support_channels")
            != [
                BASE_ASEC_SUPPORT_CHANNEL,
                PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ]
            or clone.get("puf_clone_index") != PUF_TAX_DETAIL_CLONE_INDEX
        ):
            raise ValueError(f"{boundary}: late clone support roles changed.")
        qrf_keys = {
            "seed",
            "n_estimators",
            "predictors",
            "person_outputs",
            "tax_unit_outputs",
            "invocation_mode",
            "tail_bound_quantiles",
            "worker_execution",
        }
        if not isinstance(qrf, Mapping) or set(qrf) != qrf_keys:
            raise ValueError(f"{boundary}: late primary-QRF config is malformed.")
        require_nonnegative_integer(qrf.get("seed"), label="QRF seed")
        require_positive_integer(qrf.get("n_estimators"), label="QRF n_estimators")
        invocation = qrf.get("invocation_mode")
        if not isinstance(invocation, Mapping) or set(invocation) != {
            "predictors",
            "person_outputs",
            "tax_unit_outputs",
        }:
            raise ValueError(
                f"{boundary}: late primary-QRF invocation mode is invalid."
            )
        defaults = {
            "predictors": list(PUF_TAX_DETAIL_DEFAULT_PREDICTORS),
            "person_outputs": list(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS),
            "tax_unit_outputs": list(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS),
        }
        for field, default in defaults.items():
            values = qrf.get(field)
            mode = invocation.get(field)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or mode not in {"canonical_default", "explicit"}
                or (mode == "canonical_default" and values != default)
            ):
                raise ValueError(
                    f"{boundary}: late primary-QRF {field} binding is invalid."
                )
        if qrf.get("tail_bound_quantiles") != (
            puf_tax_detail_tail_bound_quantiles_identity()
        ):
            raise ValueError(f"{boundary}: late primary-QRF tail bounds changed.")
        if qrf.get("worker_execution") != (
            _late_primary_qrf_worker_execution_binding()
        ):
            raise ValueError(f"{boundary}: late primary-QRF worker binding changed.")
        if doctrines != {
            "require_complete_recipient_predictors": True,
            "absent_cells": PUF_ABSENT_CELLS_PRESERVE_NULLS,
        }:
            raise ValueError(f"{boundary}: late primary-PUF doctrines changed.")
        if audit_sinks != {
            "fit_records": "enabled",
            "tail_bound_diagnostics": "enabled",
            "recipient_predictor_universe": "required_receipt",
        }:
            raise ValueError(f"{boundary}: late primary-PUF audit sinks changed.")
        tail_inputs = puf_capital_gains_tail_execution_inputs_identity()
        expected_tail_keys = {
            "enabled",
            "seed",
            "support_contract",
            "spec",
            "soi_e19200_agi_bands",
            "concentration_gate",
        }
        if (
            not isinstance(tail, Mapping)
            or set(tail) != expected_tail_keys
            or tail.get("enabled") is not True
            or tail.get("seed") != qrf.get("seed")
            or tail.get("support_contract")
            != puf_capital_gains_tail_support_contract_identity()
            or tail.get("spec") != puf_capital_gains_tail_spec_identity()
            or tail.get("soi_e19200_agi_bands") != tail_inputs["soi_e19200_agi_bands"]
            or tail.get("concentration_gate") != tail_inputs["concentration_gate"]
        ):
            raise ValueError(f"{boundary}: late capital-gains-tail config changed.")
        return
    if kind == "source_operator_receipt":
        require_keys({*common, "source_operator", "source_receipt_sha256"})
        expected_operator = column.removeprefix("@source_receipt:")
        if binding.get("source_operator") != expected_operator:
            raise ValueError(f"{boundary}: late source receipt owner changed.")
        _validate_sha256(
            binding.get("source_receipt_sha256"),
            boundary=f"{boundary} source receipt",
        )
        return
    if kind == "source_finalizer_execution_config":
        expected = _late_source_finalizer_execution_binding(live_runtime=False)
        require_keys(set(expected))
        if _json_ready(binding) != _json_ready(expected):
            raise ValueError(f"{boundary}: late source-finalizer config changed.")
        return
    if kind == "post_clone_source_execution_config":
        expected_operator = producer.removeprefix("source:")
        if (
            producer != f"source:{expected_operator}"
            or binding.get("operator") != expected_operator
        ):
            raise ValueError(f"{boundary}: late source execution owner changed.")
        expected = _late_source_execution_config_binding(
            producer,
            live_runtime=False,
        )
        require_keys(set(expected))
        if _json_ready(binding) != _json_ready(expected):
            raise ValueError(f"{boundary}: late source execution config changed.")
        return
    if kind == "late_transfer_model_config":
        require_keys(
            {
                *common,
                "producer",
                "entity",
                "family",
                "ordered_targets",
                "seed",
                "n_estimators",
                "max_targets_per_fit",
                "donor_spine",
                "donor_channel",
                "donor_selection",
                "donor_projection",
                "transfer_execution_contract",
            }
        )
        group = next(
            (
                item
                for item in CANONICAL_US_LATE_TRANSFER_GROUPS
                if item.name == producer
            ),
            None,
        )
        if group is None or any(
            binding.get(key) != value
            for key, value in {
                "producer": group.name,
                "entity": group.entity,
                "family": group.family,
                "ordered_targets": list(group.targets),
                "donor_spine": ASEC_PUF_DONOR_SPINE,
                "donor_channel": None,
                "donor_selection": ("all_rows_from_post_puf_asec_origin_projection"),
                "donor_projection": {
                    "support_channel": BASE_ASEC_SUPPORT_CHANNEL,
                    "support_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
                },
            }.items()
        ):
            raise ValueError(f"{boundary}: late transfer model owner/targets changed.")
        require_nonnegative_integer(binding.get("seed"), label="transfer seed")
        require_positive_integer(
            binding.get("n_estimators"), label="transfer n_estimators"
        )
        if (
            require_positive_integer(
                binding.get("max_targets_per_fit"),
                label="transfer max_targets_per_fit",
            )
            != DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
        ):
            raise ValueError(
                f"{boundary}: late transfer max_targets_per_fit is noncanonical."
            )
        if binding.get("transfer_execution_contract") != (
            acs_transfer_runtime.acs_transfer_execution_contract_identity(
                targets=group.targets,
                derive_schedule_d=False,
            )
        ):
            raise ValueError(f"{boundary}: late transfer execution contract changed.")
        return
    if kind == "late_transfer_target_bank":
        mode = binding.get("mode")
        if mode == "ephemeral_no_checkpoint":
            require_keys({*common, "mode"})
            return
        if mode == "identity_bound_checkpoint":
            require_keys({*common, "mode", "identity_sha256"})
            _validate_sha256(
                binding.get("identity_sha256"),
                boundary=f"{boundary} transfer target-bank identity",
            )
            return
        raise ValueError(f"{boundary}: late transfer target-bank mode is invalid.")
    raise AssertionError(f"Unhandled late virtual resource kind {kind!r}.")


def _late_available_input_receipt(
    *,
    producer: str,
    entity: str,
    column: str,
    rows: int,
    binding: Mapping[str, object],
) -> dict[str, object]:
    """Create one exact, hash-bound virtual-resource availability receipt."""

    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError(
            "US late-producer virtual-resource rows must be a positive integer; "
            f"got {rows!r}."
        )
    normalized_binding = _json_ready(binding)
    expected_kind = _late_virtual_resource_kind(column)
    if normalized_binding.get("resource_kind") != expected_kind:
        raise ValueError(
            f"US late-producer resource {entity}.{column} requires "
            f"resource_kind={expected_kind!r}."
        )
    expected_schema_version = _late_resource_binding_schema_version(column)
    if normalized_binding.get("schema_version") != expected_schema_version:
        raise ValueError(
            f"US late-producer resource {entity}.{column} requires binding "
            f"schema_version={expected_schema_version}."
        )
    receipt = {
        "receipt_id": f"available_input:{producer}:{entity}.{column}",
        "status": "available",
        "producer": producer,
        "entity": entity,
        "column": column,
        "rows": rows,
        "binding": normalized_binding,
        "binding_sha256": _canonical_sha256(normalized_binding),
    }
    _validate_late_available_input_receipt(
        receipt,
        producer=producer,
        entity=entity,
        column=column,
        boundary="US late-producer resource construction",
    )
    return receipt


def _validate_late_available_input_receipt(
    receipt: object,
    *,
    producer: str,
    entity: str,
    column: str,
    boundary: str,
) -> None:
    """Validate one virtual input receipt without trusting a row count alone."""

    expected_keys = {
        "receipt_id",
        "status",
        "producer",
        "entity",
        "column",
        "rows",
        "binding",
        "binding_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise ValueError(
            f"{boundary}: late-producer available-input receipt "
            f"{entity}.{column!s} does not carry its exact semantic binding."
        )
    expected = {
        "receipt_id": f"available_input:{producer}:{entity}.{column}",
        "status": "available",
        "producer": producer,
        "entity": entity,
        "column": column,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"{boundary}: late-producer available-input receipt "
            f"{entity}.{column!s} is misbound."
        )
    rows = receipt.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError(
            f"{boundary}: late-producer available-input receipt "
            f"{entity}.{column!s} has invalid rows={rows!r}."
        )
    binding = receipt.get("binding")
    expected_kind = _late_virtual_resource_kind(column)
    expected_schema_version = _late_resource_binding_schema_version(column)
    if (
        not isinstance(binding, Mapping)
        or binding.get("resource_kind") != expected_kind
        or binding.get("schema_version") != expected_schema_version
    ):
        raise ValueError(
            f"{boundary}: late-producer available-input receipt "
            f"{entity}.{column!s} has a malformed semantic binding."
        )
    binding_sha256 = receipt.get("binding_sha256")
    _validate_sha256(
        binding_sha256,
        boundary=f"{boundary} late resource {entity}.{column} binding",
    )
    if binding_sha256 != _canonical_sha256(_json_ready(binding)):
        raise ValueError(
            f"{boundary}: late-producer available-input receipt "
            f"{entity}.{column!s} binding SHA-256 mismatch."
        )
    _validate_late_resource_binding(
        binding,
        producer=producer,
        entity=entity,
        column=column,
        boundary=boundary,
    )


def _late_available_input_receipt_is_valid(
    receipt: object,
    *,
    producer: str,
    entity: str,
    column: str,
) -> bool:
    try:
        _validate_late_available_input_receipt(
            receipt,
            producer=producer,
            entity=entity,
            column=column,
            boundary="US late-producer readiness",
        )
    except (TypeError, ValueError):
        return False
    return True


def _late_string_sequence(
    values: Sequence[str] | None,
    *,
    label: str,
) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise TypeError(f"{label} must be a sequence of non-empty strings or None.")
    return list(values)


_PRIMARY_QRF_WORKER_MODULE = "microcosm.build.us_runtime.puf_qrf_worker"
_PRIMARY_QRF_SEMANTIC_ENVIRONMENT_NAMES = (
    "POPULACE_FIT_N_JOBS",
    "POPULACE_FIT_PREDICT_WORKERS",
)


def _late_primary_qrf_worker_execution_binding() -> dict[str, object]:
    """Bind the interpreter and reviewed inherited environment controls."""

    fit_jobs_raw = os.environ.get("POPULACE_FIT_N_JOBS")
    if fit_jobs_raw is None:
        fit_jobs = -1
    else:
        try:
            fit_jobs = int(fit_jobs_raw)
        except ValueError as exc:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a positive integer for the "
                "primary-QRF worker binding."
            ) from exc
        if fit_jobs < 1 or str(fit_jobs) != fit_jobs_raw:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a canonical positive integer for "
                "the primary-QRF worker binding."
            )
    predict_workers_raw = os.environ.get("POPULACE_FIT_PREDICT_WORKERS")
    if predict_workers_raw is None or not predict_workers_raw.strip():
        predict_workers = os.cpu_count() or 1
        predict_workers_source = "os_cpu_count_fallback"
    else:
        try:
            predict_workers = int(predict_workers_raw)
        except ValueError as exc:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be a positive integer for "
                "the primary-QRF worker binding."
            ) from exc
        if predict_workers < 1:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be positive for the "
                "primary-QRF worker binding."
            )
        predict_workers_source = "environment_override"
    executable = Path(sys.executable)
    return {
        "module": _PRIMARY_QRF_WORKER_MODULE,
        "argv_template": [
            str(executable),
            "-m",
            _PRIMARY_QRF_WORKER_MODULE,
            "--checkpoint-dir",
            "{checkpoint_dir}",
            "--target-index",
            "{target_index}",
        ],
        "interpreter": {
            "executable": str(executable),
            "resolved_executable": str(executable.resolve()),
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info[:3]),
        },
        "environment": {
            "policy": "inherit_parent_environment_with_bound_fit_controls",
            "overrides": {},
            "semantic_controls": {
                "POPULACE_FIT_N_JOBS": {
                    "configured": fit_jobs_raw,
                    "resolved": fit_jobs,
                },
                "POPULACE_FIT_PREDICT_WORKERS": {
                    "configured": predict_workers_raw,
                    "resolved": predict_workers,
                    "resolution": predict_workers_source,
                },
            },
            "bound_names": list(_PRIMARY_QRF_SEMANTIC_ENVIRONMENT_NAMES),
        },
    }


def _late_primary_execution_config_binding(
    *,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    seed: int,
    n_estimators: int,
    predictors: Sequence[str] | None,
    person_outputs: Sequence[str] | None,
    tax_unit_outputs: Sequence[str] | None,
    fit_records_enabled: bool,
    tail_bound_diagnostics_enabled: bool,
    capital_gains_tail_spec: PufAggregateDisaggregationSpec | None = None,
    capital_gains_tail_agi_bands: Sequence[PufE19200AgiBand] | None = None,
) -> dict[str, object]:
    """Return every non-Frame control consumed by the primary callback."""

    resolved_predictors = _late_string_sequence(
        PUF_TAX_DETAIL_DEFAULT_PREDICTORS if predictors is None else predictors,
        label="predictors",
    )
    resolved_person_outputs = _late_string_sequence(
        PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
        if person_outputs is None
        else person_outputs,
        label="person_outputs",
    )
    resolved_tax_unit_outputs = _late_string_sequence(
        PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
        if tax_unit_outputs is None
        else tax_unit_outputs,
        label="tax_unit_outputs",
    )
    tail_inputs = puf_capital_gains_tail_execution_inputs_identity(
        spec=capital_gains_tail_spec,
        agi_bands=capital_gains_tail_agi_bands,
    )
    return {
        "resource_kind": "primary_puf_execution_config",
        "schema_version": 3,
        "clone_attachment": {
            "fraction": float(clone_attachment_fraction),
            "seed": clone_attachment_seed,
            "support_channels": [
                BASE_ASEC_SUPPORT_CHANNEL,
                PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ],
            "puf_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
        },
        "qrf": {
            "seed": seed,
            "n_estimators": n_estimators,
            "predictors": resolved_predictors,
            "person_outputs": resolved_person_outputs,
            "tax_unit_outputs": resolved_tax_unit_outputs,
            "invocation_mode": {
                "predictors": (
                    "canonical_default" if predictors is None else "explicit"
                ),
                "person_outputs": (
                    "canonical_default" if person_outputs is None else "explicit"
                ),
                "tax_unit_outputs": (
                    "canonical_default" if tax_unit_outputs is None else "explicit"
                ),
            },
            "tail_bound_quantiles": (puf_tax_detail_tail_bound_quantiles_identity()),
            "worker_execution": _late_primary_qrf_worker_execution_binding(),
        },
        "doctrines": {
            "require_complete_recipient_predictors": True,
            "absent_cells": PUF_ABSENT_CELLS_PRESERVE_NULLS,
        },
        "capital_gains_tail": {
            "enabled": True,
            "seed": seed,
            "support_contract": puf_capital_gains_tail_support_contract_identity(
                capital_gains_tail_agi_bands
            ),
            "spec": tail_inputs["aggregate_disaggregation_spec"],
            "soi_e19200_agi_bands": tail_inputs["soi_e19200_agi_bands"],
            "concentration_gate": tail_inputs["concentration_gate"],
        },
        "audit_sinks": {
            "fit_records": "enabled" if fit_records_enabled else "disabled",
            "tail_bound_diagnostics": (
                "enabled" if tail_bound_diagnostics_enabled else "disabled"
            ),
            "recipient_predictor_universe": "required_receipt",
        },
    }


def _late_puf_donor_resource_semantics_binding() -> dict[str, object]:
    """Describe the dynamic donor binding without embedding build-specific bytes."""

    return {
        "resource_kind": "puf_donor_tax_units",
        "schema_version": 1,
        "runtime_identity": {
            "codec": _LATE_TABLE_DIGEST_CODEC,
            "fields": ["table_content_sha256", "ordered_columns", "dtypes"],
            "normalization": "canonical_table_string_dtypes",
        },
        "source_input_pins": ["processed_puf", "puf_source_year"],
    }


def _late_primary_qrf_checkpoint_static_binding() -> dict[str, object]:
    """Return the non-recursive primary-QRF bank semantics."""

    return {
        "resource_kind": "primary_qrf_checkpoint",
        "schema_version": 1,
        "mode": "identity_bound_checkpoint",
        "checkpoint_schema_version": PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
        "manifest_filename": PRIMARY_QRF_MANIFEST_FILENAME,
        "target_order": list(PRIMARY_QRF_TARGET_ORDER),
        "target_order_sha256": PRIMARY_QRF_TARGET_ORDER_SHA256,
    }


def stacked_late_primary_resource_receipts(
    donor_tax_units: pd.DataFrame,
    *,
    primary_qrf_checkpoint_identity_sha256: str,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    seed: int,
    n_estimators: int,
    fit_records_enabled: bool,
    tail_bound_diagnostics_enabled: bool,
    predictors: Sequence[str] | None = None,
    person_outputs: Sequence[str] | None = None,
    tax_unit_outputs: Sequence[str] | None = None,
    capital_gains_tail_spec: PufAggregateDisaggregationSpec | None = None,
    capital_gains_tail_agi_bands: Sequence[PufE19200AgiBand] | None = None,
) -> dict[str, dict[str, object]]:
    """Bind every non-Frame input consumed by the primary PUF producer."""

    if not isinstance(donor_tax_units, pd.DataFrame) or donor_tax_units.empty:
        raise ValueError("US late primary-PUF donor must be a nonempty DataFrame.")
    _validate_sha256(
        primary_qrf_checkpoint_identity_sha256,
        boundary="US late primary-QRF checkpoint identity",
    )
    if (
        isinstance(clone_attachment_fraction, bool)
        or not isinstance(clone_attachment_fraction, (int, float))
        or not np.isfinite(clone_attachment_fraction)
        or not 0 < float(clone_attachment_fraction) <= 1
    ):
        raise ValueError(
            "US late primary-PUF clone attachment fraction must be finite in "
            f"(0, 1]; got {clone_attachment_fraction!r}."
        )
    for label, value in {
        "clone_attachment_seed": clone_attachment_seed,
        "seed": seed,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"US late primary-PUF {label} must be non-negative.")
    if (
        isinstance(n_estimators, bool)
        or not isinstance(n_estimators, int)
        or n_estimators <= 0
    ):
        raise ValueError("US late primary-PUF n_estimators must be positive.")
    if not isinstance(fit_records_enabled, bool) or not isinstance(
        tail_bound_diagnostics_enabled, bool
    ):
        raise TypeError("US late primary-PUF audit-sink modes must be booleans.")
    normalized_donor = canonicalize_table_string_dtypes(
        donor_tax_units,
        boundary="late primary-PUF donor resource binding",
        table_name="puf_donor_tax_units",
    )
    donor_binding = {
        "resource_kind": "puf_donor_tax_units",
        "schema_version": 1,
        "table_content_sha256": _late_table_values_sha256(
            normalized_donor,
        ),
        "ordered_columns": [str(column) for column in normalized_donor.columns],
        "dtypes": [str(dtype) for dtype in normalized_donor.dtypes],
    }
    checkpoint_binding = {
        **_late_primary_qrf_checkpoint_static_binding(),
        "checkpoint_identity_sha256": primary_qrf_checkpoint_identity_sha256,
    }
    config_binding = _late_primary_execution_config_binding(
        clone_attachment_fraction=clone_attachment_fraction,
        clone_attachment_seed=clone_attachment_seed,
        seed=seed,
        n_estimators=n_estimators,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
        fit_records_enabled=fit_records_enabled,
        tail_bound_diagnostics_enabled=tail_bound_diagnostics_enabled,
        capital_gains_tail_spec=capital_gains_tail_spec,
        capital_gains_tail_agi_bands=capital_gains_tail_agi_bands,
    )
    return {
        "tax_unit.@puf_donor_tax_units": _late_available_input_receipt(
            producer=US_LATE_PRIMARY_PUF_STAGE,
            entity="tax_unit",
            column="@puf_donor_tax_units",
            rows=int(len(donor_tax_units)),
            binding=donor_binding,
        ),
        "tax_unit.@primary_qrf_checkpoint": _late_available_input_receipt(
            producer=US_LATE_PRIMARY_PUF_STAGE,
            entity="tax_unit",
            column="@primary_qrf_checkpoint",
            rows=1,
            binding=checkpoint_binding,
        ),
        f"tax_unit.{US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT}": (
            _late_available_input_receipt(
                producer=US_LATE_PRIMARY_PUF_STAGE,
                entity="tax_unit",
                column=US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT,
                rows=1,
                binding=config_binding,
            )
        ),
    }


def _validate_stacked_late_primary_checkpoint_input_binding(
    binding: object,
    *,
    boundary: str,
) -> None:
    """Authenticate the sidecar that prevents stale primary-QRF bank reuse."""

    expected_keys = {
        "artifact_kind",
        "schema_version",
        "primary_resource_receipts",
        "sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        raise ValueError(f"{boundary}: primary-QRF input binding schema drifted.")
    if (
        binding.get("artifact_kind") != _LATE_PRIMARY_QRF_INPUT_BINDING_ARTIFACT_KIND
        or binding.get("schema_version") != 1
    ):
        raise ValueError(f"{boundary}: primary-QRF input binding identity changed.")
    resources = binding.get("primary_resource_receipts")
    contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[US_LATE_PRIMARY_PUF_STAGE]
    expected_resources = _late_contract_available_input_keys(contract)
    if not isinstance(resources, Mapping) or set(resources) != expected_resources:
        raise ValueError(
            f"{boundary}: primary-QRF input binding does not cover its exact "
            "declared resource surface."
        )
    for key, receipt in resources.items():
        entity, column = key.split(".", 1)
        _validate_late_available_input_receipt(
            receipt,
            producer=US_LATE_PRIMARY_PUF_STAGE,
            entity=entity,
            column=column,
            boundary=boundary,
        )
    unsigned = dict(binding)
    sha256 = unsigned.pop("sha256")
    _validate_sha256(sha256, boundary=f"{boundary} primary-QRF input binding")
    if sha256 != _canonical_sha256(_json_ready(unsigned)):
        raise ValueError(f"{boundary}: primary-QRF input binding SHA-256 mismatch.")


def stacked_late_primary_checkpoint_input_binding(
    primary_resource_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build the durable input sidecar for the primary-QRF checkpoint bank."""

    resources = {
        key: _json_ready(receipt)
        for key, receipt in sorted(primary_resource_receipts.items())
    }
    payload: dict[str, object] = {
        "artifact_kind": _LATE_PRIMARY_QRF_INPUT_BINDING_ARTIFACT_KIND,
        "schema_version": 1,
        "primary_resource_receipts": resources,
    }
    payload["sha256"] = _canonical_sha256(payload)
    _validate_stacked_late_primary_checkpoint_input_binding(
        payload,
        boundary="US stacked late primary-QRF input-binding construction",
    )
    return payload


_SOURCE_MANIFEST_STAGE_BY_OPERATOR: Mapping[str, str] = MappingProxyType(
    {
        "with_us_prior_year_income_inputs": "prior_year_income",
        "with_us_medicare_take_up_input": "medicare_take_up_input",
        "with_us_pregnancy_inputs": "pregnancy",
        "with_us_wic_claim_input": "wic_claim_input",
        "with_us_child_support_inputs": "child_support_inputs",
        "with_us_disability_benefits": "disability_benefits_input",
        "with_us_workers_compensation": "workers_compensation_input",
        "with_us_weeks_unemployed": "weeks_unemployed_input",
        "with_us_childcare_inputs": "childcare_inputs",
        "with_us_adult_care_inputs": "adult_care_inputs",
        "with_us_energy_subsidy_input": "energy_subsidy",
        "with_us_retirement_contribution_inputs": "retirement_contributions",
        "with_us_retirement_distribution_inputs": "retirement_distributions",
        "with_us_immigration_inputs": "immigration_status",
        "with_us_education_inputs": "education_inputs",
    }
)
_SOURCE_STAGE_SPEC_RESOLVER_BY_OPERATOR: Mapping[str, tuple[str, str]] = (
    MappingProxyType(
        {
            "with_us_prior_year_income_inputs": (
                "microcosm.build.us_runtime.prior_year_income",
                "us_prior_year_income_stage_spec",
            ),
            "with_us_medicare_take_up_input": (
                "microcosm.build.us_runtime.medicare_take_up",
                "us_medicare_take_up_stage_spec",
            ),
            "with_us_pregnancy_inputs": (
                "microcosm.build.us_runtime.pregnancy",
                "us_pregnancy_stage_spec",
            ),
            "with_us_wic_claim_input": (
                "microcosm.build.us_runtime.wic_claim",
                "us_wic_claim_stage_spec",
            ),
            "with_us_child_support_inputs": (
                "microcosm.build.us_runtime.child_support",
                "us_child_support_stage_spec",
            ),
            "with_us_disability_benefits": (
                "microcosm.build.us_runtime.disability_benefits",
                "us_disability_benefits_stage_spec",
            ),
            "with_us_workers_compensation": (
                "microcosm.build.us_runtime.workers_compensation",
                "us_workers_compensation_stage_spec",
            ),
            "with_us_weeks_unemployed": (
                "microcosm.build.us_runtime.weeks_unemployed",
                "us_weeks_unemployed_stage_spec",
            ),
            "with_us_childcare_inputs": (
                "microcosm.build.us_runtime.childcare",
                "us_childcare_stage_spec",
            ),
            "with_us_adult_care_inputs": (
                "microcosm.build.us_runtime.adult_care",
                "us_adult_care_stage_spec",
            ),
            "with_us_energy_subsidy_input": (
                "microcosm.build.us_runtime.energy_subsidy",
                "us_energy_subsidy_stage_spec",
            ),
            "with_us_retirement_contribution_inputs": (
                "microcosm.build.us_runtime.retirement_contributions",
                "us_retirement_contributions_stage_spec",
            ),
            "with_us_retirement_distribution_inputs": (
                "microcosm.build.us_runtime.retirement_distributions",
                "us_retirement_distributions_stage_spec",
            ),
            "with_us_immigration_inputs": (
                "microcosm.build.us_runtime.immigration",
                "us_immigration_stage_spec",
            ),
            "with_us_education_inputs": (
                "microcosm.build.us_runtime.education_inputs",
                "us_education_inputs_stage_spec",
            ),
        }
    )
)
_DIRECT_HOUSING_ASSISTANCE_SOURCE_OPERATOR = (
    "impute_us_housing_assistance_to_puf_support"
)
if len(_SOURCE_MANIFEST_STAGE_BY_OPERATOR) != 15 or set(
    _SOURCE_MANIFEST_STAGE_BY_OPERATOR
) | {_DIRECT_HOUSING_ASSISTANCE_SOURCE_OPERATOR} != set(
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
):
    raise RuntimeError(
        "US late source callback-identity map must cover exactly fifteen "
        "manifest stages plus the direct housing-assistance QRF."
    )
if set(_SOURCE_STAGE_SPEC_RESOLVER_BY_OPERATOR) != set(
    _SOURCE_MANIFEST_STAGE_BY_OPERATOR
):
    raise RuntimeError(
        "US late source runtime-spec resolver map must cover exactly the "
        "fifteen manifest-backed callbacks."
    )


def _late_source_stage_spec_binding(
    operator: str,
    *,
    resource: object | None = None,
) -> dict[str, object] | None:
    """Resolve the exact packaged SourceStageSpec consumed by a callback."""

    if operator == _DIRECT_HOUSING_ASSISTANCE_SOURCE_OPERATOR:
        return None
    try:
        stage_name = _SOURCE_MANIFEST_STAGE_BY_OPERATOR[operator]
    except KeyError as exc:
        raise ValueError(
            f"US late source operator {operator!r} has no manifest-stage binding."
        ) from exc
    resolved_resource = (
        files("microcosm.build.us").joinpath("source_stages.json")
        if resource is None
        else resource
    )
    if not hasattr(resolved_resource, "read_bytes"):
        raise TypeError("US source-stage binding resource must expose read_bytes().")
    manifest = load_source_manifest(resolved_resource)
    stage_map = manifest.stage_map()
    if stage_name not in stage_map:
        raise ValueError(
            f"US source manifest declares no bound stage {stage_name!r} for "
            f"operator {operator!r}."
        )
    stage_spec = _json_ready(asdict(stage_map[stage_name]))
    resolver_module, resolver_name = _SOURCE_STAGE_SPEC_RESOLVER_BY_OPERATOR[operator]
    runtime_spec_verified = resource is None
    if runtime_spec_verified:
        runtime_module = importlib.import_module(resolver_module)
        runtime_resolver = getattr(runtime_module, resolver_name, None)
        if not callable(runtime_resolver):
            raise RuntimeError(
                f"US late source runtime resolver {resolver_module}.{resolver_name} "
                "is unavailable."
            )
        runtime_stage_spec = _json_ready(asdict(runtime_resolver()))
        if runtime_stage_spec != stage_spec:
            raise ValueError(
                f"US late source operator {operator!r} runtime SourceStageSpec "
                "differs from the content-bound packaged manifest."
            )
    asset_bytes = resolved_resource.read_bytes()
    return {
        "asset": "microcosm.build.us/source_stages.json",
        "asset_sha256": hashlib.sha256(asset_bytes).hexdigest(),
        "manifest": {
            "country": manifest.country,
            "version": manifest.version,
            "policy": manifest.policy,
        },
        "stage_name": stage_name,
        "resolved_stage_spec": stage_spec,
        "resolved_stage_spec_sha256": _canonical_sha256(stage_spec),
        "runtime_stage_spec_resolver": {
            "module": resolver_module,
            "callable": resolver_name,
        },
        "runtime_stage_spec_verified": runtime_spec_verified,
    }


def _late_source_execution_config_binding(
    producer_name: str,
    *,
    live_runtime: bool = True,
) -> dict[str, object]:
    """Return all fixed controls consumed by one post-clone source callback."""

    operator = producer_name.removeprefix("source:")
    if producer_name != f"source:{operator}":
        raise ValueError(
            f"US late source producer name is malformed: {producer_name!r}."
        )
    if live_runtime:
        seed = multispine_pool_runtime.POOL_RANDOM_SEED
        time_period = multispine_pool_runtime.POOL_TIME_PERIOD
        allow_existing = (
            multispine_pool_runtime.POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE
        )
        housing_n_estimators = (
            multispine_pool_runtime.POOL_HOUSING_ASSISTANCE_N_ESTIMATORS
        )
        housing_max_train_samples = (
            multispine_pool_runtime.POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES
        )
        operator_registry = (
            multispine_pool_runtime.POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        )
        phase = multispine_pool_runtime._POST_CLONE_PHASE
        operator_contracts = multispine_pool_runtime.POOL_OPERATOR_CONTRACTS
        output_families = multispine_pool_runtime.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES
        formula_owned_outputs = multispine_pool_runtime._FORMULA_OWNED_SOURCE_OUTPUTS
    else:
        seed = POOL_RANDOM_SEED
        time_period = POOL_TIME_PERIOD
        allow_existing = POOL_SOURCE_ALLOW_EXISTING_WITHOUT_SOURCE
        housing_n_estimators = POOL_HOUSING_ASSISTANCE_N_ESTIMATORS
        housing_max_train_samples = POOL_HOUSING_ASSISTANCE_MAX_TRAIN_SAMPLES
        operator_registry = POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        phase = POOL_POST_CLONE_SOURCE_PHASE
        operator_contracts = POOL_OPERATOR_CONTRACTS
        output_families = PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES
        formula_owned_outputs = FORMULA_OWNED_SOURCE_COLUMNS
    try:
        operator_contract = operator_contracts[operator]
    except KeyError as exc:
        raise ValueError(
            f"US late source runtime declares no operator {operator!r}."
        ) from exc
    family_outputs = output_families[operator_contract.family]
    return {
        "resource_kind": "post_clone_source_execution_config",
        "schema_version": 3,
        "operator": operator,
        "phase": phase,
        "operator_registry": list(operator_registry),
        "operator_contract": {
            "family": operator_contract.family,
            "phases": list(operator_contract.phases),
            "mechanism": operator_contract.mechanism,
            "execution_scope": operator_contract.execution_scope,
        },
        "declared_output_family": {
            entity: sorted(columns)
            for entity, columns in sorted(family_outputs.items())
        },
        "formula_owned_outputs_removed": {
            entity: sorted(set(columns) & set(formula_owned_outputs.get(entity, ())))
            for entity, columns in sorted(family_outputs.items())
            if set(columns) & set(formula_owned_outputs.get(entity, ()))
        },
        "seed": seed,
        "time_period": (
            None
            if operator == "impute_us_housing_assistance_to_puf_support"
            else time_period
        ),
        "force_puf_imputation": (
            True if operator == "with_us_retirement_distribution_inputs" else None
        ),
        "allow_existing_without_source": (
            allow_existing
            if operator
            in {
                "with_us_child_support_inputs",
                "with_us_disability_benefits",
                "with_us_workers_compensation",
                "with_us_childcare_inputs",
                "with_us_adult_care_inputs",
                "with_us_energy_subsidy_input",
            }
            else None
        ),
        "housing_assistance_qrf": (
            {
                "n_estimators": housing_n_estimators,
                "max_train_samples": housing_max_train_samples,
            }
            if operator == _DIRECT_HOUSING_ASSISTANCE_SOURCE_OPERATOR
            else None
        ),
        "external_sidecars": (
            {"asec_2023_source": {"mode": "not_supplied"}}
            if operator == "with_us_weeks_unemployed"
            else {"asec_education_source": {"mode": "not_supplied"}}
            if operator == "with_us_education_inputs"
            else {}
        ),
        "source_stage_spec": _late_source_stage_spec_binding(operator),
    }


def _late_source_resource_receipts(
    *,
    producer_name: str,
) -> dict[str, dict[str, object]]:
    """Bind the fixed controls consumed by one post-clone source callback."""

    binding = _late_source_execution_config_binding(producer_name)
    return {
        f"person.{US_LATE_SOURCE_EXECUTION_CONFIG_INPUT}": (
            _late_available_input_receipt(
                producer=producer_name,
                entity="person",
                column=US_LATE_SOURCE_EXECUTION_CONFIG_INPUT,
                rows=1,
                binding=binding,
            )
        )
    }


def _late_source_finalizer_execution_binding(
    *,
    live_runtime: bool = True,
) -> dict[str, object]:
    """Bind every doctrine input consumed by the source finalizer callback."""

    if live_runtime:
        phase = multispine_pool_runtime._POST_CLONE_PHASE
        operator_registry = (
            multispine_pool_runtime.POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        )
        formula_owned_outputs = multispine_pool_runtime._FORMULA_OWNED_SOURCE_OUTPUTS
        deferred_inputs = multispine_pool_runtime.POOL_DEFERRED_TRANSFER_INPUTS
        deferred_status = multispine_pool_runtime.POOL_DEFERRED_TRANSFER_STATUS
    else:
        phase = POOL_POST_CLONE_SOURCE_PHASE
        operator_registry = POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        formula_owned_outputs = FORMULA_OWNED_SOURCE_COLUMNS
        deferred_inputs = POOL_DEFERRED_TRANSFER_INPUTS
        deferred_status = POOL_DEFERRED_TRANSFER_STATUS
    return {
        "resource_kind": "source_finalizer_execution_config",
        "schema_version": 2,
        "phase": phase,
        "source_operator_registry": list(operator_registry),
        "formula_owned_output_exclusions": {
            entity: sorted(columns)
            for entity, columns in sorted(formula_owned_outputs.items())
        },
        "deferred_transfer_inputs": _json_ready(deferred_inputs),
        "deferred_status": deferred_status,
    }


def _late_source_finalizer_resource_receipts() -> dict[str, dict[str, object]]:
    """Bind the finalizer's phase, registry, exclusions, and deferral contract."""

    column = US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT
    return {
        f"person.{column}": _late_available_input_receipt(
            producer=US_LATE_SOURCE_FINALIZER_STAGE,
            entity="person",
            column=column,
            rows=1,
            binding=_late_source_finalizer_execution_binding(),
        )
    }


def _assert_late_callback_consumed_bound_config(
    *,
    producer: str,
    entity: str,
    column: str,
    available_input_receipts: Mapping[str, Mapping[str, object]],
    actual_binding: Mapping[str, object],
) -> None:
    """Refuse a callback whose live runtime config differs from its DAG input."""

    key = f"{entity}.{column}"
    receipt = available_input_receipts.get(key)
    bound = receipt.get("binding") if isinstance(receipt, Mapping) else None
    if not isinstance(bound, Mapping) or _json_ready(bound) != _json_ready(
        actual_binding
    ):
        raise ValueError(
            f"Late producer {producer!r} consumed a runtime execution config "
            f"that differs from its bound input {key}."
        )


_CANONICAL_ACS_EARNINGS_UNIVERSE_CONTRACT_IDENTITY = dict(
    acs_pums_earnings_universe_contract_identity()
)


def _late_acs_earnings_universe_execution_binding() -> dict[str, object]:
    """Return the exact rules and scope consumed by the universe producer."""

    return {
        "resource_kind": "acs_pums_earnings_universe_execution_config",
        "schema_version": 2,
        "runtime_identity_owner": ("microcosm.build.us_runtime.acs_income_universe"),
        "ordered_mapped_columns": list(
            acs_income_universe_runtime.ACS_PUMS_EARNINGS_SOURCE_COLUMNS
        ),
        "person_scope_mode": "whole_frame_acs_channel",
        "contract_identity": (
            acs_income_universe_runtime.acs_pums_earnings_universe_contract_identity()
        ),
    }


def _late_acs_earnings_universe_resource_receipts() -> dict[str, dict[str, object]]:
    """Bind the exact rules and scope consumed by the universe producer."""

    column = US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT
    return {
        f"person.{column}": _late_available_input_receipt(
            producer=US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
            entity="person",
            column=column,
            rows=1,
            binding=_late_acs_earnings_universe_execution_binding(),
        )
    }


def _late_transfer_model_config_binding(
    *,
    group_name: str,
    entity: str,
    family: str,
    targets: Sequence[str],
    seed: int,
    n_estimators: int,
    max_targets_per_fit: int,
) -> dict[str, object]:
    """Return every static model and donor-selection control for one group."""

    return {
        "resource_kind": "late_transfer_model_config",
        "schema_version": 3,
        "producer": group_name,
        "entity": entity,
        "family": family,
        "ordered_targets": list(targets),
        "seed": seed,
        "n_estimators": n_estimators,
        "max_targets_per_fit": max_targets_per_fit,
        "donor_spine": ASEC_PUF_DONOR_SPINE,
        "donor_channel": None,
        "donor_selection": "all_rows_from_post_puf_asec_origin_projection",
        "donor_projection": {
            "support_channel": BASE_ASEC_SUPPORT_CHANNEL,
            "support_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
        },
        "transfer_execution_contract": (
            acs_transfer_runtime.acs_transfer_execution_contract_identity(
                targets=targets,
                derive_schedule_d=False,
            )
        ),
    }


def _late_transfer_target_bank_static_binding(*, mode: str) -> dict[str, object]:
    """Return the shared non-recursive target-bank mode binding."""

    if mode not in {"ephemeral_no_checkpoint", "identity_bound_checkpoint"}:
        raise ValueError(f"Unsupported US late target-bank mode {mode!r}.")
    return {
        "resource_kind": "late_transfer_target_bank",
        "schema_version": 1,
        "mode": mode,
    }


def _late_transfer_target_bank_binding(
    *,
    group_name: str,
    target_bank: AcsTransferTargetBank | None,
) -> dict[str, object]:
    """Return one runtime target-bank binding without hiding its mode."""

    if target_bank is None:
        return _late_transfer_target_bank_static_binding(mode="ephemeral_no_checkpoint")
    identity_sha256 = getattr(target_bank, "identity_sha256", None)
    _validate_sha256(
        identity_sha256,
        boundary=f"US late transfer {group_name!r} target-bank identity",
    )
    return {
        **_late_transfer_target_bank_static_binding(mode="identity_bound_checkpoint"),
        "identity_sha256": identity_sha256,
    }


def _late_transfer_resource_receipts(
    *,
    group_name: str,
    entity: str,
    family: str,
    targets: Sequence[str],
    seed: int,
    n_estimators: int,
    max_targets_per_fit: int,
    target_bank: AcsTransferTargetBank | None,
) -> dict[str, dict[str, object]]:
    """Bind model controls and durable-bank identity for one transfer node."""

    model_binding = _late_transfer_model_config_binding(
        group_name=group_name,
        entity=entity,
        family=family,
        targets=targets,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
    )
    bank_binding = _late_transfer_target_bank_binding(
        group_name=group_name,
        target_bank=target_bank,
    )
    return {
        f"{entity}.{US_LATE_TRANSFER_MODEL_CONFIG_INPUT}": (
            _late_available_input_receipt(
                producer=group_name,
                entity=entity,
                column=US_LATE_TRANSFER_MODEL_CONFIG_INPUT,
                rows=1,
                binding=model_binding,
            )
        ),
        f"{entity}.{US_LATE_TRANSFER_TARGET_BANK_INPUT}": (
            _late_available_input_receipt(
                producer=group_name,
                entity=entity,
                column=US_LATE_TRANSFER_TARGET_BANK_INPUT,
                rows=1,
                binding=bank_binding,
            )
        ),
    }


def stacked_late_producer_resource_semantics_receipt(
    *,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    primary_seed: int,
    primary_n_estimators: int,
    transfer_seed: int,
    transfer_n_estimators: int,
    transfer_max_targets_per_fit: int,
) -> dict[str, object]:
    """Bind every static or derivation-mode resource in the late DAG.

    Runtime table and bank digests remain dynamic. Their exact codecs and
    derivations are bound here so the outer checkpoint identity covers the
    complete resource doctrine without recursively embedding its own digest.
    """

    schedule_receipt = us_late_producer_schedule_receipt()
    group_by_name = {group.name: group for group in CANONICAL_US_LATE_TRANSFER_GROUPS}
    producer_rows: list[dict[str, object]] = []
    for producer_name in CANONICAL_US_LATE_PRODUCER_SCHEDULE.order:
        contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[producer_name]
        expected_keys = _late_contract_available_input_keys(contract)
        resources: dict[str, object]
        if contract.kind == "acs_earnings_universe":
            resources = {
                f"person.{US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT}": {
                    "resolution": "static_exact",
                    "binding": _late_acs_earnings_universe_execution_binding(),
                }
            }
        elif contract.kind == "primary_puf":
            resources = {
                "tax_unit.@puf_donor_tax_units": {
                    "resolution": "runtime_content_bound",
                    "binding": _late_puf_donor_resource_semantics_binding(),
                },
                "tax_unit.@primary_qrf_checkpoint": {
                    "resolution": "outer_identity_derived",
                    "binding": _late_primary_qrf_checkpoint_static_binding(),
                    "dynamic_field": {
                        "name": "checkpoint_identity_sha256",
                        "derivation": (
                            "canonical_sha256(outer_stacked_checkpoint_base_identity)"
                        ),
                    },
                },
                f"tax_unit.{US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT}": {
                    "resolution": "static_exact",
                    "binding": _late_primary_execution_config_binding(
                        clone_attachment_fraction=clone_attachment_fraction,
                        clone_attachment_seed=clone_attachment_seed,
                        seed=primary_seed,
                        n_estimators=primary_n_estimators,
                        predictors=None,
                        person_outputs=None,
                        tax_unit_outputs=None,
                        fit_records_enabled=True,
                        tail_bound_diagnostics_enabled=True,
                    ),
                },
            }
        elif contract.kind == "post_clone_source":
            resources = {
                f"person.{US_LATE_SOURCE_EXECUTION_CONFIG_INPUT}": {
                    "resolution": "static_exact",
                    "binding": _late_source_execution_config_binding(producer_name),
                }
            }
        elif contract.kind == "source_finalizer":
            resources = {
                f"person.{US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT}": {
                    "resolution": "static_exact",
                    "binding": _late_source_finalizer_execution_binding(),
                }
            }
            resources.update(
                {
                    f"person.@source_receipt:{operator}": {
                        "resolution": "signed_callback_receipt_derived",
                        "binding": {
                            "resource_kind": "source_operator_receipt",
                            "schema_version": 1,
                            "source_operator": operator,
                            "dynamic_field": {
                                "name": "source_receipt_sha256",
                                "derivation": (
                                    "canonical_sha256(signed_source_callback_receipt)"
                                ),
                            },
                        },
                    }
                    for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
                }
            )
        elif contract.kind == "late_transfer":
            group = group_by_name[producer_name]
            resources = {
                f"{group.entity}.{US_LATE_TRANSFER_MODEL_CONFIG_INPUT}": {
                    "resolution": "static_exact",
                    "binding": _late_transfer_model_config_binding(
                        group_name=group.name,
                        entity=group.entity,
                        family=group.family,
                        targets=group.targets,
                        seed=transfer_seed,
                        n_estimators=transfer_n_estimators,
                        max_targets_per_fit=transfer_max_targets_per_fit,
                    ),
                },
                f"{group.entity}.{US_LATE_TRANSFER_TARGET_BANK_INPUT}": {
                    "resolution": "transferred_stage_identity_derived",
                    "binding": _late_transfer_target_bank_static_binding(
                        mode="identity_bound_checkpoint"
                    ),
                    "dynamic_field": {
                        "name": "identity_sha256",
                        "derivation": {
                            "base": "transferred_pool_checkpoint_stage_identity",
                            "late_producer_dag_sha256": schedule_receipt[
                                "schedule_sha256"
                            ],
                            "late_producer_schedule_sha256": schedule_receipt[
                                "payload_sha256"
                            ],
                            "producer": {
                                "name": group.name,
                                "entity": group.entity,
                                "family": group.family,
                                "ordered_targets": list(group.targets),
                            },
                        },
                    },
                },
            }
        else:  # pragma: no cover - import-validated canonical kind partition
            raise AssertionError(f"Unhandled late producer kind {contract.kind!r}.")
        if set(resources) != expected_keys:
            raise RuntimeError(
                f"US late resource semantics for {producer_name!r} do not cover "
                f"its exact virtual surface; missing={sorted(expected_keys - set(resources))}, "
                f"extra={sorted(set(resources) - expected_keys)}."
            )
        producer_rows.append(
            {
                "producer": producer_name,
                "kind": contract.kind,
                "resources": {
                    key: _json_ready(resources[key]) for key in sorted(resources)
                },
            }
        )
    payload: dict[str, object] = {
        "artifact_kind": _LATE_RESOURCE_SEMANTICS_ARTIFACT_KIND,
        "schema_version": 1,
        "producer_schedule_sha256": schedule_receipt["schedule_sha256"],
        "producer_schedule_payload_sha256": schedule_receipt["payload_sha256"],
        "producer_count": len(producer_rows),
        "producers": producer_rows,
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _late_frame_content_sha256(frame: Frame) -> str:
    """Hash a live frame while excluding the self-referential authority key."""

    metadata = {
        key: value
        for key, value in frame.metadata.items()
        if key != US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY
    }
    table_receipts = {
        name: _late_table_values_sha256(
            frame.table(name),
            normalize_strings=True,
        )
        for name in frame.entities
    }
    link_receipts = {
        name: _late_table_values_sha256(
            frame.link(name),
            normalize_strings=True,
        )
        for name in frame.links
    }
    weight_receipts = {}
    for entity in frame.weighted_entities:
        weights = frame.weights_for(entity)
        digest = hashlib.sha256(weights.values.astype("<f8", copy=False).tobytes())
        weight_receipts[entity] = {
            "kind": weights.kind.value,
            "rows": len(weights.values),
            "sha256": digest.hexdigest(),
        }
    payload = {
        "version": 1,
        "entities": list(frame.entities),
        "links": list(frame.links),
        "tables": table_receipts,
        "link_tables": link_receipts,
        "weights": weight_receipts,
        "strata_sha256": _late_table_values_sha256(
            frame.strata.rename("stratum").to_frame(),
            normalize_strings=True,
        ),
        "mass_log": [
            {
                "entity": record.entity,
                "old_total": record.old_total,
                "new_total": record.new_total,
                "declared_factor": record.declared_factor,
                "reason": record.reason,
            }
            for record in frame.mass_log
        ],
        "metadata": _json_ready(metadata),
    }
    return _canonical_sha256(payload)


def _late_input_column_evidence(
    frame: Frame,
    *,
    input_column: ProducerInputColumn,
    required_scope: str,
    producer_name: str,
    available_input_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Bind one declared physical or virtual input to its live content."""

    missing_rows, invalid_rows = _late_input_column_readiness_rows(
        frame,
        input_column=input_column,
        required_scope=required_scope,
        producer_name=producer_name,
        available_input_receipts=available_input_receipts,
    )
    payload: dict[str, object] = {
        "entity": input_column.entity,
        "column": input_column.column,
        "value_kind": input_column.value_kind,
        "missing_rows": missing_rows,
        "invalid_rows": invalid_rows,
    }
    if input_column.entity == "frame":
        metadata_key = input_column.column.removeprefix("@")
        value = frame.metadata.get(metadata_key)
        payload.update(
            {
                "required_scope": required_scope,
                "scope_rows": 1,
                "status": "present" if isinstance(value, Mapping) else "absent",
                "content_sha256": _canonical_sha256(
                    _json_ready(value)
                    if isinstance(value, Mapping)
                    else {"absent": True}
                ),
            }
        )
        return payload
    table = frame.table(input_column.entity)
    scope = _late_required_scope_mask(
        frame,
        entity=input_column.entity,
        required_scope=required_scope,
    )
    payload.update(
        {
            "required_scope": required_scope,
            "scope_rows": int(scope.sum()),
        }
    )
    if input_column.column == "@resolved_weight":
        weights = frame.resolve_weights(input_column.entity)
        scoped = pd.DataFrame(
            {"resolved_weight": weights.values[scope.to_numpy(dtype=bool)]},
            index=table.index[scope],
        )
        payload.update(
            {
                "status": "present",
                "weight_kind": weights.kind.value,
                "content_sha256": _late_table_values_sha256(scoped),
            }
        )
        return payload
    if input_column.column.startswith("@"):
        receipt_key = f"{input_column.entity}.{input_column.column}"
        receipt = available_input_receipts.get(receipt_key)
        payload.update(
            {
                "status": "present" if isinstance(receipt, Mapping) else "absent",
                "content_sha256": _canonical_sha256(
                    _json_ready(receipt)
                    if isinstance(receipt, Mapping)
                    else {"absent": True}
                ),
            }
        )
        return payload
    if input_column.column not in table:
        payload.update(
            {
                "status": "absent",
                "content_sha256": _canonical_sha256({"absent": True}),
            }
        )
        return payload
    values = table.loc[scope, [input_column.column]]
    payload.update(
        {
            "status": "present",
            "content_sha256": _late_table_values_sha256(values),
        }
    )
    return payload


def _late_declared_input_evidence(
    frame: Frame,
    contract: ProducerContract,
    *,
    available_input_receipts: Mapping[str, Mapping[str, object]],
    unfilled_rows: Mapping[ProducerInput, int],
    invalid_rows: Mapping[ProducerInput, int],
) -> list[dict[str, object]]:
    """Build the exact, content-bound readiness surface for one producer."""

    result: list[dict[str, object]] = []
    for requirement in contract.inputs:
        evidence = {
            "alternatives": [
                [
                    _late_input_column_evidence(
                        frame,
                        input_column=column,
                        required_scope=requirement.required_scope,
                        producer_name=contract.name,
                        available_input_receipts=available_input_receipts,
                    )
                    for column in alternative
                ]
                for alternative in requirement.alternatives
            ]
        }
        evidence["sha256"] = _canonical_sha256(evidence)
        result.append(
            {
                "entity": requirement.entity,
                "column": requirement.column,
                "required_scope": requirement.required_scope,
                "producing_stage": requirement.producing_stage,
                "unfilled_rows": unfilled_rows[requirement],
                "invalid_rows": invalid_rows[requirement],
                "evidence": evidence,
            }
        )
    return result


def _late_scalar_is_hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _late_output_matches_metric_family(values: pd.Series, metric: str) -> bool:
    """Return whether one late output's physical dtype matches its registry family."""

    is_boolean = pd.api.types.is_bool_dtype(values.dtype)
    is_real_numeric = bool(
        pd.api.types.is_numeric_dtype(values.dtype)
        and not is_boolean
        and not pd.api.types.is_complex_dtype(values.dtype)
    )
    if metric == "boolean_incidence":
        return bool(is_boolean)
    if metric in {"monetary_sign_separated", "rare_incidence"}:
        return is_real_numeric
    if metric != "categorical_tvd":
        return False
    if is_boolean or pd.api.types.is_complex_dtype(values.dtype):
        return False
    if is_real_numeric or isinstance(values.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_object_dtype(values.dtype):
        inferred = pd.api.types.infer_dtype(values, skipna=True)
        return inferred != "boolean" and all(
            _late_scalar_is_hashable(value) for value in values.dropna()
        )
    return bool(pd.api.types.is_string_dtype(values.dtype))


def _validate_late_callback_output_metric_families(
    frame: Frame,
    contract: ProducerContract,
    *,
    metric_registry: Mapping[
        tuple[str, str, str, int], str
    ] = _CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY_ANCHOR,
) -> None:
    """Fail closed when a callback materializes a registered output incorrectly."""

    declared_by_column: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (entity, family, column, clone_index), metric in metric_registry.items():
        if clone_index != 0:
            continue
        declared_by_column.setdefault((entity, column), []).append((family, metric))

    failures: list[str] = []
    for output in contract.outputs:
        if output.entity == "frame" or output.column.startswith("@"):
            continue
        declarations = declared_by_column.get((output.entity, output.column), ())
        if not declarations:
            continue
        metrics = {metric for _family, metric in declarations}
        if len(metrics) != 1:
            raise RuntimeError(
                "The origin-battery metric registry gives conflicting dtype "
                f"families to {output.entity}.{output.column}: {declarations}."
            )
        table = frame.table(output.entity)
        if output.column not in table:
            continue
        metric = next(iter(metrics))
        values = table[output.column]
        if not _late_output_matches_metric_family(values, metric):
            families = sorted(family for family, _metric in declarations)
            failures.append(
                f"{output.entity}.{output.column}: registry families {families} "
                f"declare {metric!r}, callback dtype is {str(values.dtype)!r}"
            )
    if failures:
        raise TypeError(
            f"Late producer {contract.name!r} output dtype-family validation "
            "failed:\n  " + "\n  ".join(failures)
        )


def _late_output_column_evidence(
    frame: Frame,
    *,
    output: ProducerOutput,
    producer_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Bind one declared producer output to its post-callback content."""

    payload: dict[str, object] = {
        "entity": output.entity,
        "column": output.column,
        "coverage_scope": output.coverage_scope,
    }
    if output.entity == "frame":
        value = frame.metadata.get(output.column.removeprefix("@"))
        payload.update(
            {
                "status": "present" if isinstance(value, Mapping) else "absent",
                "content_sha256": _canonical_sha256(
                    _json_ready(value)
                    if isinstance(value, Mapping)
                    else {"absent": True}
                ),
            }
        )
        return payload
    table = frame.table(output.entity)
    if output.column.startswith("@source_receipt:"):
        payload.update(
            {
                "scope_rows": len(table),
                "status": "present",
                "content_sha256": _canonical_sha256(_json_ready(producer_receipt)),
            }
        )
        return payload
    scope = _late_required_scope_mask(
        frame,
        entity=output.entity,
        required_scope=output.coverage_scope,
    )
    payload["scope_rows"] = int(scope.sum())
    if output.column == "@resolved_weight":
        weights = frame.resolve_weights(output.entity)
        scoped = pd.DataFrame(
            {"resolved_weight": weights.values[scope.to_numpy(dtype=bool)]},
            index=table.index[scope],
        )
        payload.update(
            {
                "status": "present",
                "weight_kind": weights.kind.value,
                "content_sha256": _late_table_values_sha256(scoped),
            }
        )
    elif output.column not in table:
        payload.update(
            {
                "status": "absent",
                "content_sha256": _canonical_sha256({"absent": True}),
            }
        )
    else:
        payload.update(
            {
                "status": "present",
                "content_sha256": _late_table_values_sha256(
                    table.loc[scope, [output.column]]
                ),
            }
        )
    return payload


def _late_producer_transition_authority_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Derive the immutable live-frame anchor for one signed DAG receipt."""

    schedule = receipt["producer_schedule"]
    assert isinstance(schedule, Mapping)
    authority: dict[str, object] = {
        "authority_id": US_LATE_PRODUCER_TRANSITION_AUTHORITY_ID,
        "version": US_LATE_PRODUCER_TRANSITION_AUTHORITY_VERSION,
        "receipt_sha256": receipt["sha256"],
        "producer_schedule_sha256": schedule["payload_sha256"],
        "input_frame_sha256": receipt["input_frame_sha256"],
        "output_frame_sha256": receipt["output_frame_sha256"],
        "execution_chain_sha256": receipt["execution_chain_sha256"],
    }
    authority["sha256"] = _canonical_sha256(authority)
    return authority


def _bind_late_producer_transition_authority(
    frame: Frame,
    receipt: Mapping[str, object],
) -> tuple[Frame, str]:
    """Bind a generated DAG transition into immutable Frame metadata."""

    if US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY in frame.metadata:
        raise ValueError(
            "US late-producer transition authority is already bound; refusing "
            "to overwrite the immutable generation anchor."
        )
    authority = _late_producer_transition_authority_receipt(receipt)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({link: frame.link(link).copy() for link in frame.links})
    bound = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY: authority,
        },
    )
    return bound, str(authority["sha256"])


def _validate_primary_callback_resource_binding(
    producer_receipt: Mapping[str, object],
    *,
    available_input_receipts: Mapping[str, object],
    boundary: str,
) -> None:
    """Prove the primary callback consumed the resources gated by its DAG row."""

    observed = producer_receipt.get("primary_resource_receipts_sha256")
    _validate_sha256(observed, boundary=f"{boundary} primary callback resources")
    expected = _canonical_sha256(_json_ready(available_input_receipts))
    if observed != expected:
        raise ValueError(
            f"{boundary}: primary callback receipt disagrees with its declared "
            f"resources; expected={expected}, observed={observed}."
        )


def _validate_late_execution_row(
    raw_row: object,
    *,
    contract: ProducerContract,
    execution_index: int,
    expected_previous_sha256: str,
    boundary: str,
) -> str:
    """Re-run one persisted readiness proof without invoking its callback."""

    if not isinstance(raw_row, Mapping):
        raise ValueError(
            f"{boundary}: late producer execution row {execution_index} is not "
            "an object."
        )
    expected_keys = {
        "execution_index",
        "producer",
        "kind",
        "declared_inputs",
        "declared_absence_receipts",
        "available_input_receipts",
        "input_surface_sha256",
        "output_surface",
        "output_surface_sha256",
        "producer_receipt",
        "producer_receipt_sha256",
        "previous_execution_sha256",
        "status",
        "sha256",
    }
    if set(raw_row) != expected_keys:
        raise ValueError(
            f"{boundary}: late producer execution row {execution_index} schema "
            f"drifted; missing={sorted(expected_keys - set(raw_row))}, "
            f"extra={sorted(set(raw_row) - expected_keys)}."
        )
    expected_status = "complete"
    if (
        raw_row.get("execution_index") != execution_index
        or raw_row.get("producer") != contract.name
        or raw_row.get("kind") != contract.kind
        or raw_row.get("status") != expected_status
    ):
        raise ValueError(
            f"{boundary}: late producer execution row {execution_index} is "
            f"misbound; expected producer={contract.name!r}, kind={contract.kind!r}, "
            f"status={expected_status!r}."
        )
    declared_inputs = raw_row.get("declared_inputs")
    if not isinstance(declared_inputs, list) or len(declared_inputs) != len(
        contract.inputs
    ):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} does not carry its "
            f"exact {len(contract.inputs)}-input readiness surface."
        )
    unfilled_rows: dict[ProducerInput, int] = {}
    invalid_rows: dict[ProducerInput, int] = {}
    evidenced_available_keys: set[str] = set()
    evidenced_available_sha256: dict[str, str] = {}
    physical_input_states: dict[
        tuple[str, str, str], tuple[int, int, str, str, str | None]
    ] = {}
    scope_cardinalities: dict[tuple[str, str], int] = {}
    for requirement, raw_input in zip(contract.inputs, declared_inputs, strict=True):
        if not isinstance(raw_input, Mapping):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} has a malformed "
                "declared-input receipt."
            )
        expected_input = {
            "entity": requirement.entity,
            "column": requirement.column,
            "required_scope": requirement.required_scope,
            "producing_stage": requirement.producing_stage,
        }
        expected_input_keys = {
            *expected_input,
            "unfilled_rows",
            "invalid_rows",
            "evidence",
        }
        if set(raw_input) != expected_input_keys:
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input receipt "
                f"schema drifted from {requirement.entity}.{requirement.column}."
            )
        if any(raw_input.get(key) != value for key, value in expected_input.items()):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input receipt "
                f"drifted from {requirement.entity}.{requirement.column}."
            )
        rows = raw_input.get("unfilled_rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input "
                f"{requirement.entity}.{requirement.column} has invalid "
                f"unfilled_rows={rows!r}."
            )
        unfilled_rows[requirement] = rows
        invalid = raw_input.get("invalid_rows")
        if isinstance(invalid, bool) or not isinstance(invalid, int) or invalid < 0:
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input "
                f"{requirement.entity}.{requirement.column} has invalid "
                f"invalid_rows={invalid!r}."
            )
        invalid_rows[requirement] = invalid
        evidence = raw_input.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "alternatives",
            "sha256",
        }:
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input "
                f"{requirement.entity}.{requirement.column} has malformed "
                "content evidence."
            )
        evidence_unsigned = dict(evidence)
        evidence_sha256 = evidence_unsigned.pop("sha256")
        _validate_sha256(
            evidence_sha256,
            boundary=(f"{boundary} late producer {contract.name!r} input evidence"),
        )
        if evidence_sha256 != _canonical_sha256(evidence_unsigned):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input "
                f"{requirement.entity}.{requirement.column} evidence SHA-256 "
                "mismatch."
            )
        alternatives = evidence.get("alternatives")
        if not isinstance(alternatives, list) or len(alternatives) != len(
            requirement.alternatives
        ):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input "
                f"{requirement.entity}.{requirement.column} evidence changed "
                "its alternative surface."
            )
        column_states: dict[
            ProducerInputColumn, tuple[int, int, dict[str, object]]
        ] = {}
        alternative_missing_counts: list[int] = []
        for declared_alternative, raw_alternative in zip(
            requirement.alternatives,
            alternatives,
            strict=True,
        ):
            if not isinstance(raw_alternative, list) or len(raw_alternative) != len(
                declared_alternative
            ):
                raise ValueError(
                    f"{boundary}: late producer {contract.name!r} input "
                    f"{requirement.entity}.{requirement.column} evidence changed "
                    "one physical alternative."
                )
            alternative_missing = 0
            for declared_column, raw_column in zip(
                declared_alternative,
                raw_alternative,
                strict=True,
            ):
                if not isinstance(raw_column, Mapping) or any(
                    raw_column.get(field) != value
                    for field, value in {
                        "entity": declared_column.entity,
                        "column": declared_column.column,
                        "value_kind": declared_column.value_kind,
                        "required_scope": requirement.required_scope,
                    }.items()
                ):
                    raise ValueError(
                        f"{boundary}: late producer {contract.name!r} input "
                        f"{requirement.entity}.{requirement.column} evidence "
                        "is misbound to its physical columns."
                    )
                expected_column_keys = {
                    "entity",
                    "column",
                    "value_kind",
                    "required_scope",
                    "scope_rows",
                    "missing_rows",
                    "invalid_rows",
                    "status",
                    "content_sha256",
                }
                if declared_column.column == "@resolved_weight":
                    expected_column_keys.add("weight_kind")
                if set(raw_column) != expected_column_keys:
                    raise ValueError(
                        f"{boundary}: late producer {contract.name!r} input "
                        f"{requirement.entity}.{requirement.column} evidence "
                        f"schema drifted for {declared_column.entity}."
                        f"{declared_column.column}."
                    )
                for count_field in ("scope_rows", "missing_rows", "invalid_rows"):
                    count = raw_column.get(count_field)
                    if (
                        isinstance(count, bool)
                        or not isinstance(count, int)
                        or count < 0
                    ):
                        raise ValueError(
                            f"{boundary}: late producer {contract.name!r} "
                            f"input evidence has invalid {count_field}={count!r}."
                        )
                scope_rows = int(raw_column["scope_rows"])
                missing_rows = int(raw_column["missing_rows"])
                evidence_invalid_rows = int(raw_column["invalid_rows"])
                status = raw_column.get("status")
                if status not in {"present", "absent"}:
                    raise ValueError(
                        f"{boundary}: late producer {contract.name!r} input "
                        f"evidence has invalid status={status!r}."
                    )
                is_virtual = (
                    declared_column.column.startswith("@")
                    and declared_column.column != "@resolved_weight"
                    and declared_column.entity != "frame"
                )
                if declared_column.entity == "frame" and scope_rows != 1:
                    raise ValueError(
                        f"{boundary}: late producer {contract.name!r} frame "
                        "input evidence must have scope_rows=1."
                    )
                if status == "absent":
                    expected_missing = (
                        1
                        if declared_column.entity == "frame"
                        else max(1, scope_rows)
                        if is_virtual
                        else scope_rows
                    )
                    if (
                        missing_rows != expected_missing
                        or evidence_invalid_rows != 0
                        or raw_column.get("content_sha256")
                        != _canonical_sha256({"absent": True})
                    ):
                        raise ValueError(
                            f"{boundary}: late producer {contract.name!r} "
                            "absent input evidence has inconsistent counts or "
                            "content identity."
                        )
                elif declared_column.entity == "frame" or is_virtual:
                    if missing_rows != 0 or evidence_invalid_rows != 0:
                        raise ValueError(
                            f"{boundary}: late producer {contract.name!r} "
                            "present virtual input evidence has nonzero "
                            "readiness counts."
                        )
                elif declared_column.column == "@resolved_weight":
                    if (
                        missing_rows != 0
                        or not isinstance(raw_column.get("weight_kind"), str)
                        or not raw_column.get("weight_kind")
                    ):
                        raise ValueError(
                            f"{boundary}: late producer {contract.name!r} "
                            "resolved-weight evidence is malformed."
                        )
                elif (
                    missing_rows > scope_rows
                    or evidence_invalid_rows > scope_rows
                    or missing_rows + evidence_invalid_rows > scope_rows
                ):
                    raise ValueError(
                        f"{boundary}: late producer {contract.name!r} physical "
                        "input evidence counts exceed its declared scope."
                    )
                _validate_sha256(
                    raw_column.get("content_sha256"),
                    boundary=(
                        f"{boundary} late producer {contract.name!r} input content"
                    ),
                )
                normalized_column = dict(raw_column)
                if (
                    declared_column.entity != "frame"
                    and not declared_column.column.startswith("@")
                ):
                    physical_key = (
                        declared_column.entity,
                        declared_column.column,
                        requirement.required_scope,
                    )
                    physical_state = (
                        scope_rows,
                        missing_rows,
                        str(status),
                        str(raw_column["content_sha256"]),
                        str(raw_column["weight_kind"])
                        if "weight_kind" in raw_column
                        else None,
                    )
                    previous_physical_state = physical_input_states.get(physical_key)
                    if (
                        previous_physical_state is not None
                        and previous_physical_state != physical_state
                    ):
                        raise ValueError(
                            f"{boundary}: late producer {contract.name!r} carries "
                            "inconsistent physical input evidence for "
                            f"{declared_column.entity}.{declared_column.column} "
                            f"on {requirement.required_scope}."
                        )
                    physical_input_states[physical_key] = physical_state
                    scope_key = (
                        declared_column.entity,
                        requirement.required_scope,
                    )
                    previous_scope_rows = scope_cardinalities.get(scope_key)
                    if (
                        previous_scope_rows is not None
                        and previous_scope_rows != scope_rows
                    ):
                        raise ValueError(
                            f"{boundary}: late producer {contract.name!r} has "
                            "inconsistent input scope cardinality for "
                            f"{declared_column.entity}.{requirement.required_scope}."
                        )
                    scope_cardinalities[scope_key] = scope_rows
                previous_state = column_states.get(declared_column)
                current_state = (
                    missing_rows,
                    evidence_invalid_rows,
                    normalized_column,
                )
                if previous_state is not None and previous_state != current_state:
                    raise ValueError(
                        f"{boundary}: late producer {contract.name!r} repeats "
                        "one physical input with inconsistent evidence."
                    )
                column_states[declared_column] = current_state
                alternative_missing += missing_rows
                if (
                    declared_column.column.startswith("@")
                    and declared_column.column != "@resolved_weight"
                    and declared_column.entity != "frame"
                    and raw_column.get("status") == "present"
                ):
                    evidence_key = f"{declared_column.entity}.{declared_column.column}"
                    evidenced_available_keys.add(evidence_key)
                    evidenced_available_sha256[evidence_key] = str(
                        raw_column["content_sha256"]
                    )
            alternative_missing_counts.append(alternative_missing)
        recomputed_unfilled = min(alternative_missing_counts)
        recomputed_invalid = sum(state[1] for state in column_states.values())
        if rows != recomputed_unfilled or invalid != recomputed_invalid:
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} input "
                f"{requirement.entity}.{requirement.column} readiness counts "
                "disagree with its physical evidence; "
                f"declared=({rows}, {invalid}), "
                f"recomputed=({recomputed_unfilled}, {recomputed_invalid})."
            )

    raw_absence = raw_row.get("declared_absence_receipts")
    if not isinstance(raw_absence, Mapping):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} absence receipts are "
            "not an object."
        )
    expected_absence_ids = {
        receipt_id
        for requirement, rows in unfilled_rows.items()
        if rows > 0 and invalid_rows[requirement] == 0
        for receipt_id in requirement.tolerated_absence_receipts
    }
    if set(raw_absence) != expected_absence_ids:
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} declared-absence "
            f"surface drifted; expected={sorted(expected_absence_ids)}, "
            f"got={sorted(map(str, raw_absence))}."
        )
    for requirement, rows in unfilled_rows.items():
        if rows <= 0:
            continue
        for receipt_id in requirement.tolerated_absence_receipts:
            receipt = raw_absence.get(receipt_id)
            expected_receipt = {
                "receipt_id": receipt_id,
                "status": "declared_absence",
                "entity": requirement.entity,
                "column": requirement.column,
                "required_scope": requirement.required_scope,
                "rows": rows,
                "producer": contract.name,
                "reason": "optional availability-pattern input",
            }
            if not isinstance(receipt, Mapping) or dict(receipt) != expected_receipt:
                raise ValueError(
                    f"{boundary}: late producer {contract.name!r} absence receipt "
                    f"{receipt_id!r} is not canonical."
                )

    available_inputs = raw_row.get("available_input_receipts")
    if not isinstance(available_inputs, Mapping):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} available-input "
            "receipts are not an object."
        )
    if contract.kind in {
        "acs_earnings_universe",
        "primary_puf",
        "source_finalizer",
        "late_transfer",
    }:
        mandatory_available_keys = _late_contract_available_input_keys(contract)
    elif contract.kind == "post_clone_source":
        mandatory_available_keys = {f"person.{US_LATE_SOURCE_EXECUTION_CONFIG_INPUT}"}
    else:
        mandatory_available_keys = set()
    if not mandatory_available_keys <= evidenced_available_keys:
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} virtual-input "
            "evidence does not prove every mandatory available resource."
        )
    if contract.kind in {
        "acs_earnings_universe",
        "primary_puf",
        "source_finalizer",
        "late_transfer",
    }:
        if evidenced_available_keys != mandatory_available_keys:
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} virtual-input "
                "evidence adds a noncanonical mandatory resource."
            )
    expected_available_keys = evidenced_available_keys
    if set(available_inputs) != expected_available_keys:
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} available-input "
            f"surface drifted; expected={sorted(expected_available_keys)}, "
            f"got={sorted(map(str, available_inputs))}."
        )
    for key, receipt in available_inputs.items():
        entity, column = key.split(".", 1)
        _validate_late_available_input_receipt(
            receipt,
            producer=contract.name,
            entity=entity,
            column=column,
            boundary=f"{boundary} late producer {contract.name!r}",
        )
        if evidenced_available_sha256.get(key) != _canonical_sha256(
            _json_ready(receipt)
        ):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} available-input "
                f"receipt {key!r} disagrees with its declared content evidence."
            )

    input_surface_sha256 = raw_row.get("input_surface_sha256")
    _validate_sha256(
        input_surface_sha256,
        boundary=f"{boundary} late producer {contract.name!r} input surface",
    )
    if input_surface_sha256 != _canonical_sha256(declared_inputs):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} input-surface "
            "SHA-256 mismatch."
        )

    output_surface = raw_row.get("output_surface")
    if not isinstance(output_surface, list) or len(output_surface) != len(
        contract.outputs
    ):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} does not carry its "
            f"exact {len(contract.outputs)}-output content surface."
        )
    for output, raw_output in zip(contract.outputs, output_surface, strict=True):
        expected_output = {
            "entity": output.entity,
            "column": output.column,
            "coverage_scope": output.coverage_scope,
        }
        expected_output_keys = {
            *expected_output,
            "status",
            "content_sha256",
        }
        if output.entity != "frame":
            expected_output_keys.add("scope_rows")
        if output.column == "@resolved_weight":
            expected_output_keys.add("weight_kind")
        if (
            not isinstance(raw_output, Mapping)
            or set(raw_output) != expected_output_keys
            or any(
                raw_output.get(field) != value
                for field, value in expected_output.items()
            )
        ):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} output evidence "
                f"drifted from {output.entity}.{output.column}."
            )
        if raw_output.get("status") != "present":
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} completed with "
                f"absent output {output.entity}.{output.column}."
            )
        if output.entity != "frame":
            scope_rows = raw_output.get("scope_rows")
            if (
                isinstance(scope_rows, bool)
                or not isinstance(scope_rows, int)
                or scope_rows < 0
            ):
                raise ValueError(
                    f"{boundary}: late producer {contract.name!r} output "
                    f"{output.entity}.{output.column} has invalid "
                    f"scope_rows={scope_rows!r}."
                )
            if contract.kind != "primary_puf" and not output.column.startswith("@"):
                scope_key = (output.entity, output.coverage_scope)
                previous_scope_rows = scope_cardinalities.get(scope_key)
                if (
                    previous_scope_rows is not None
                    and previous_scope_rows != scope_rows
                ):
                    raise ValueError(
                        f"{boundary}: late producer {contract.name!r} output "
                        "scope cardinality disagrees with its declared input "
                        f"scope for {output.entity}.{output.coverage_scope}; "
                        f"input={previous_scope_rows}, output={scope_rows}."
                    )
                scope_cardinalities[scope_key] = int(scope_rows)
        if output.column == "@resolved_weight" and (
            not isinstance(raw_output.get("weight_kind"), str)
            or not raw_output.get("weight_kind")
        ):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} resolved-weight "
                "output evidence is malformed."
            )
        _validate_sha256(
            raw_output.get("content_sha256"),
            boundary=f"{boundary} late producer {contract.name!r} output content",
        )
    output_surface_sha256 = raw_row.get("output_surface_sha256")
    _validate_sha256(
        output_surface_sha256,
        boundary=f"{boundary} late producer {contract.name!r} output surface",
    )
    if output_surface_sha256 != _canonical_sha256(output_surface):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} output-surface "
            "SHA-256 mismatch."
        )

    producer_receipt = raw_row.get("producer_receipt")
    if not isinstance(producer_receipt, Mapping):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} callback receipt is "
            "not an object."
        )
    producer_receipt_sha256 = raw_row.get("producer_receipt_sha256")
    _validate_sha256(
        producer_receipt_sha256,
        boundary=f"{boundary} late producer {contract.name!r} callback receipt",
    )
    if producer_receipt_sha256 != _canonical_sha256(producer_receipt):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} callback-receipt "
            "SHA-256 mismatch."
        )
    for raw_output in output_surface:
        if str(raw_output["column"]).startswith("@source_receipt:") and (
            raw_output["content_sha256"] != producer_receipt_sha256
        ):
            raise ValueError(
                f"{boundary}: late producer {contract.name!r} source-receipt "
                "output digest disagrees with its callback receipt."
            )
    if contract.kind == "primary_puf":
        _validate_primary_callback_resource_binding(
            producer_receipt,
            available_input_receipts=available_inputs,
            boundary=f"{boundary} late producer {contract.name!r}",
        )

    if raw_row.get("previous_execution_sha256") != expected_previous_sha256:
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} execution chain does "
            "not name the preceding digest."
        )

    run_producer_when_ready(
        contract,
        lambda: None,
        unfilled_rows=unfilled_rows,
        invalid_rows=invalid_rows,
        absence_receipts=raw_absence,
    )
    unsigned = dict(raw_row)
    observed_sha256 = unsigned.pop("sha256")
    _validate_sha256(
        observed_sha256,
        boundary=f"{boundary} late producer {contract.name!r} execution row",
    )
    if observed_sha256 != _canonical_sha256(unsigned):
        raise ValueError(
            f"{boundary}: late producer {contract.name!r} execution-row "
            "SHA-256 mismatch."
        )
    return str(observed_sha256)


def _late_execution_genesis_sha256(
    *,
    producer_schedule_sha256: object,
    input_frame_sha256: object,
) -> str:
    return _canonical_sha256(
        {
            "receipt_schema_version": US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
            "producer_schedule_sha256": producer_schedule_sha256,
            "input_frame_sha256": input_frame_sha256,
        }
    )


def _validate_late_transition_authority(
    frame: Frame,
    receipt: Mapping[str, object],
    *,
    boundary: str,
    expected_transition_authority_sha256: str,
    require_live_output: bool,
) -> None:
    """Authenticate the immutable frame anchor and independent digest carrier."""

    _validate_sha256(
        expected_transition_authority_sha256,
        boundary=f"{boundary} independently carried late-producer authority",
    )
    authority = frame.metadata.get(US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY)
    if not isinstance(authority, Mapping):
        raise ValueError(
            f"{boundary}: late-producer transition authority is absent from "
            "live frame metadata."
        )
    expected_authority = _late_producer_transition_authority_receipt(receipt)
    if dict(authority) != expected_authority:
        raise ValueError(
            f"{boundary}: late-producer receipt is not bound to the immutable "
            "live transition authority."
        )
    if authority.get("sha256") != expected_transition_authority_sha256:
        raise ValueError(
            f"{boundary}: late-producer receipt differs from the independently "
            "carried late-producer transition authority."
        )
    if require_live_output:
        live_output_sha256 = _late_frame_content_sha256(frame)
        if receipt.get("output_frame_sha256") != live_output_sha256:
            raise ValueError(
                f"{boundary}: late-producer output digest does not match the "
                "live frame."
            )


def validate_stacked_late_producer_transition_authority(
    frame: Frame,
    receipt: Mapping[str, object],
    *,
    boundary: str,
    expected_transition_authority_sha256: str,
) -> None:
    """Validate the anchor after declared downstream operators have run."""

    validate_stacked_late_producer_receipt(receipt, boundary=boundary)
    _validate_late_transition_authority(
        frame,
        receipt,
        boundary=boundary,
        expected_transition_authority_sha256=(expected_transition_authority_sha256),
        require_live_output=False,
    )


def validate_stacked_late_producer_receipt(
    receipt: Mapping[str, object],
    *,
    boundary: str,
    frame: Frame | None = None,
    expected_transition_authority_sha256: str | None = None,
) -> None:
    """Authenticate the complete derived execution and source/transfer proof."""

    if not isinstance(receipt, Mapping):
        raise ValueError(f"{boundary}: stacked late-producer DAG receipt is absent.")
    expected_keys = {
        "version",
        "producer_schedule",
        "input_frame_sha256",
        "output_frame_sha256",
        "execution_chain_sha256",
        "execution",
        "source_completion",
        "post_puf_transfer",
        "sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError(
            f"{boundary}: stacked late-producer DAG receipt schema drifted; "
            f"missing={sorted(expected_keys - set(receipt))}, "
            f"extra={sorted(set(receipt) - expected_keys)}."
        )
    if receipt.get("version") != US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"{boundary}: stacked late-producer DAG receipt version changed."
        )
    expected_schedule = _json_ready(us_late_producer_schedule_receipt())
    schedule = receipt.get("producer_schedule")
    if not isinstance(schedule, Mapping) or _json_ready(schedule) != expected_schedule:
        raise ValueError(
            f"{boundary}: stacked late-producer DAG schedule is not canonical."
        )
    for digest_field in (
        "input_frame_sha256",
        "output_frame_sha256",
        "execution_chain_sha256",
        "sha256",
    ):
        _validate_sha256(
            receipt.get(digest_field), boundary=f"{boundary} {digest_field}"
        )
    execution = receipt.get("execution")
    expected_order = CANONICAL_US_LATE_PRODUCER_SCHEDULE.order
    if not isinstance(execution, list) or len(execution) != len(expected_order):
        raise ValueError(
            f"{boundary}: stacked late-producer DAG must carry exactly "
            f"{len(expected_order)} execution rows."
        )
    previous_sha256 = _late_execution_genesis_sha256(
        producer_schedule_sha256=schedule["payload_sha256"],
        input_frame_sha256=receipt["input_frame_sha256"],
    )
    execution_by_name: dict[str, Mapping[str, object]] = {}
    for index, producer_name in enumerate(expected_order):
        raw_row = execution[index]
        previous_sha256 = _validate_late_execution_row(
            raw_row,
            contract=CANONICAL_US_LATE_PRODUCER_REGISTRY[producer_name],
            execution_index=index,
            expected_previous_sha256=previous_sha256,
            boundary=boundary,
        )
        assert isinstance(raw_row, Mapping)
        execution_by_name[producer_name] = raw_row
    if receipt.get("execution_chain_sha256") != previous_sha256:
        raise ValueError(
            f"{boundary}: stacked late-producer execution-chain terminus "
            "does not match its final row."
        )

    expected_source_order = [
        producer.removeprefix("source:")
        for producer in expected_order
        if producer.startswith("source:")
    ]
    source_completion = receipt.get("source_completion")
    if not isinstance(source_completion, Mapping):
        raise ValueError(
            f"{boundary}: stacked late-producer DAG source completion is absent."
        )
    suboperators = source_completion.get("suboperators")
    if (
        source_completion.get("phase") != "post_clone"
        or source_completion.get("operator_order") != expected_source_order
        or not isinstance(suboperators, list)
        or len(suboperators) != len(expected_source_order)
    ):
        raise ValueError(
            f"{boundary}: stacked late-producer DAG source completion does not "
            "match the derived sixteen-source order."
        )
    for index, (operator, suboperator) in enumerate(
        zip(expected_source_order, suboperators, strict=True)
    ):
        if (
            not isinstance(suboperator, Mapping)
            or suboperator.get("operator") != operator
            or suboperator.get("order_index") != index
        ):
            raise ValueError(
                f"{boundary}: stacked source completion row {index} is not "
                f"bound to {operator!r}."
            )
    deferred = source_completion.get("deferred_transfer_inputs")
    deferred_inputs = deferred.get("inputs") if isinstance(deferred, Mapping) else None
    if not isinstance(deferred_inputs, Mapping) or set(deferred_inputs) != {
        "bank_account_assets",
        "bond_assets",
        "stock_assets",
    }:
        raise ValueError(
            f"{boundary}: stacked source completion lacks the exact three-input "
            "deferred-source receipt."
        )

    normalized_suboperators: list[dict[str, object]] = []
    source_evidence: list[object] = []
    for index, operator in enumerate(expected_source_order):
        source_row = execution_by_name[f"source:{operator}"]
        source_receipt = source_row.get("producer_receipt")
        if not isinstance(source_receipt, Mapping):
            raise ValueError(
                f"{boundary}: source producer {operator!r} has no callback receipt."
            )
        source_suboperators = source_receipt.get("suboperators")
        if (
            source_receipt.get("phase") != "post_clone"
            or source_receipt.get("operator_order") != [operator]
            or not isinstance(source_suboperators, list)
            or len(source_suboperators) != 1
            or not isinstance(source_suboperators[0], Mapping)
            or source_suboperators[0].get("operator") != operator
        ):
            raise ValueError(
                f"{boundary}: source producer {operator!r} callback receipt is "
                "not the canonical single-operator proof."
            )
        normalized = dict(source_suboperators[0])
        normalized["order_index"] = index
        normalized_suboperators.append(normalized)
        source_evidence.append(source_receipt.get("cps_source_evidence"))
    if suboperators != normalized_suboperators:
        raise ValueError(
            f"{boundary}: stacked source completion is not reconstructed from "
            "the sixteen producer receipts."
        )
    if source_evidence and any(
        evidence != source_evidence[0] for evidence in source_evidence[1:]
    ):
        raise ValueError(
            f"{boundary}: stacked source producer receipts disagree on CPS "
            "source evidence."
        )
    if source_completion.get("cps_source_evidence") != (
        source_evidence[0] if source_evidence else None
    ):
        raise ValueError(
            f"{boundary}: stacked source completion CPS evidence is not bound "
            "to its producer receipts."
        )
    finalizer_row = execution_by_name[US_LATE_SOURCE_FINALIZER_STAGE]
    finalizer_inputs = finalizer_row.get("available_input_receipts")
    assert isinstance(finalizer_inputs, Mapping)
    for operator in expected_source_order:
        key = f"person.@source_receipt:{operator}"
        source_receipt = execution_by_name[f"source:{operator}"]["producer_receipt"]
        input_receipt = finalizer_inputs.get(key)
        input_binding = (
            input_receipt.get("binding") if isinstance(input_receipt, Mapping) else None
        )
        if not isinstance(input_binding, Mapping) or input_binding.get(
            "source_receipt_sha256"
        ) != _canonical_sha256(source_receipt):
            raise ValueError(
                f"{boundary}: source finalizer input {key!r} is not bound to "
                "its producer receipt."
            )
    if _json_ready(finalizer_row["producer_receipt"]) != _json_ready(source_completion):
        raise ValueError(
            f"{boundary}: source-finalizer execution receipt differs from the "
            "aggregate source completion proof."
        )

    transfer = receipt.get("post_puf_transfer")
    if not isinstance(transfer, Mapping):
        raise ValueError(
            f"{boundary}: stacked late-producer DAG transfer proof is absent."
        )
    validate_stacked_post_puf_transfer_receipt(transfer, boundary=boundary)
    groups = transfer["groups"]
    assert isinstance(groups, Mapping)
    canonical_family = {
        (entity, target): family
        for entity, families in CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        for family, targets in families.items()
        for target in targets
    }
    reconstructed_targets: dict[str, object] = {}
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        row_receipt = execution_by_name[group.name]["producer_receipt"]
        if _json_ready(row_receipt) != _json_ready(groups[group.name]):
            raise ValueError(
                f"{boundary}: transfer execution receipt for {group.name!r} "
                "differs from its aggregate group proof."
            )
        assert isinstance(row_receipt, Mapping)
        group_targets = row_receipt.get("targets")
        assert isinstance(group_targets, Mapping)
        for target in group.targets:
            bounded_label = f"{group.entity}/{group.family}/{target}"
            aggregate_label = (
                f"{group.entity}/{canonical_family[(group.entity, target)]}/{target}"
            )
            reconstructed_targets[aggregate_label] = group_targets[bounded_label]
    if _json_ready(transfer["targets"]) != _json_ready(reconstructed_targets):
        raise ValueError(
            f"{boundary}: aggregate late-transfer targets are not reconstructed "
            "from the nineteen producer receipts."
        )

    unsigned = dict(receipt)
    observed_sha256 = unsigned.pop("sha256")
    if observed_sha256 != _canonical_sha256(unsigned):
        raise ValueError(
            f"{boundary}: stacked late-producer DAG receipt SHA-256 mismatch."
        )
    if (frame is None) != (expected_transition_authority_sha256 is None):
        raise ValueError(
            f"{boundary}: live frame and independently carried late-producer "
            "authority must be supplied together."
        )
    if frame is not None:
        assert expected_transition_authority_sha256 is not None
        _validate_late_transition_authority(
            frame,
            receipt,
            boundary=boundary,
            expected_transition_authority_sha256=(expected_transition_authority_sha256),
            require_live_output=True,
        )


def _validate_test_authority(authority: _StackedAuthority, *, boundary: str) -> None:
    """Keep the explicit fixture seam visibly and terminally non-production."""

    receipt = _authority_receipt(authority)
    if (
        authority.declared_form != _NONCANONICAL_AUTHORITY_FORM
        or receipt["authority_form"] != _NONCANONICAL_AUTHORITY_FORM
        or receipt["canonical_identity"] is not False
    ):
        raise ValueError(f"{boundary} requires a NON-CANONICAL test authority.")


def _validate_stacked_gate_manifest_details(
    gate_name: str,
    details: Mapping[str, object],
    *,
    passed: bool,
    _canonical_surface: TargetFamilies = CANONICAL_STACKED_DECLARED_SURFACE,
    _canonical_plan: tuple[GapFillDirection, ...] = CANONICAL_STACKED_GAP_FILL_PLAN,
    _canonical_registry: Mapping[
        tuple[str, str, str, int], str
    ] = CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY,
    _canonical_joint_registry: Mapping[
        tuple[str, str, tuple[str, ...], int], str
    ] = CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY,
) -> None:
    """Validate gate-specific receipts against the captured canonical doctrine."""

    boundary = f"Gate {gate_name!r} manifest emission"
    authority = details.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError(
            f"{boundary}: no stacked authority receipt; production manifest "
            "emission is forbidden."
        )
    _validate_production_authority_receipt(authority, boundary=boundary)
    authority_sha256 = authority["sha256"]
    plan_sha256 = authority["components"]["gap_fill_plan"]["sha256"]
    post_puf_surface_sha256 = authority["components"]["post_puf_transfer_surface"][
        "sha256"
    ]
    surface_sha256 = authority["components"]["declared_surface"]["sha256"]
    expected_keys = _surface_target_keys(_canonical_surface)

    def reject(reason: str) -> None:
        raise ValueError(
            f"{boundary}: {reason}; production manifest emission is forbidden."
        )

    tail_support_receipt = details.get(_TAIL_SUPPORT_GATE_DETAIL_KEY)
    if tail_support_receipt is not None:
        if not isinstance(tail_support_receipt, Mapping):
            reject("capital-gains-tail support receipt must be an object")
        try:
            validate_puf_capital_gains_tail_terminal_support_receipt(
                tail_support_receipt
            )
        except (TypeError, ValueError) as error:
            reject(f"capital-gains-tail support receipt is invalid: {error}")

    def nonnegative_int(
        receipt: Mapping[str, object],
        field_name: str,
        *,
        label: str,
    ) -> int:
        value = receipt.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            reject(f"{label} {field_name} must be a non-negative integer")
        return value

    def validate_structural_absence_receipt(
        raw_receipt: object,
        *,
        label: str,
        battery: bool,
    ) -> tuple[int, dict[str, int]]:
        if not isinstance(raw_receipt, Mapping):
            reject(f"{label} must carry canonical recipient-absence authority")
        expected_fields = {
            "rule_id",
            "selection",
            "reason",
            "status",
            "rows",
            "by_origin_role",
            "unexpected_null_rows",
            "structural_rows_filled",
        }
        if battery:
            expected_fields.update(
                {"comparison_clone_index", "rows_excluded_from_scope"}
            )
        if set(raw_receipt) != expected_fields:
            reject(f"{label} structural-absence receipt schema mismatch")
        if (
            raw_receipt.get("rule_id") != _ACS_GQ_RENT_ABSENCE_RULE_ID
            or raw_receipt.get("selection") != _ACS_GQ_RENT_ABSENCE_SELECTION
            or raw_receipt.get("reason") != _ACS_GQ_RENT_ABSENCE_REASON
            or raw_receipt.get("status") != "exact_structural_absence"
        ):
            reject(f"{label} structural-absence doctrine mismatch")
        rows = nonnegative_int(raw_receipt, "rows", label=label)
        unexpected = nonnegative_int(
            raw_receipt,
            "unexpected_null_rows",
            label=label,
        )
        filled = nonnegative_int(
            raw_receipt,
            "structural_rows_filled",
            label=label,
        )
        raw_by_role = raw_receipt.get("by_origin_role")
        if not isinstance(raw_by_role, Mapping):
            reject(f"{label} structural by-origin-role counts are not a mapping")
        by_role: dict[str, int] = {}
        for cell, count in raw_by_role.items():
            cell_channel, separator, clone_role = str(cell).partition("/clone_")
            if (
                not separator
                or cell_channel != ACS_STACKED_SUPPORT_CHANNEL
                or not clone_role.isdigit()
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                reject(f"{label} structural by-origin-role receipt is malformed")
            by_role[str(cell)] = count
        if rows != sum(by_role.values()):
            reject(f"{label} structural row count does not equal its role counts")
        if passed and (unexpected != 0 or filled != 0):
            reject(f"{label} passing structural-absence equation is not exact")
        if battery:
            if raw_receipt.get("comparison_clone_index") != 0:
                reject(f"{label} structural battery clone scope must be clone 0")
            excluded = nonnegative_int(
                raw_receipt,
                "rows_excluded_from_scope",
                label=label,
            )
            if excluded != by_role.get(f"{ACS_STACKED_SUPPORT_CHANNEL}/clone_0", 0):
                reject(
                    f"{label} structural battery exclusion count differs from "
                    "its clone-0 authority"
                )
        return rows, by_role

    if gate_name == _COMPLETENESS_GATE_NAME:
        expected_labels = {
            f"{entity}/{family}/{column}"
            for entity, family, column, _clone_index in expected_keys
        }
        direction_by_label = {
            f"{entity}/{family}/{column}": direction
            for direction in _canonical_plan
            for entity, families in direction.target_families.items()
            for family, columns in families.items()
            for column in columns
        }
        targets = details.get("targets")
        if details.get("declared_targets") != len(expected_keys):
            reject(
                "canonical completeness receipt must declare exactly "
                f"{len(expected_keys)} targets"
            )
        if not isinstance(targets, Mapping) or set(targets) != expected_labels:
            reject("canonical completeness receipt target surface mismatch")
        allowed_target_forms = {
            "observed_complete",
            "invalid_declared_metric_values",
            "missing_declared_entity",
            "missing_declared_target",
            "origin_exact_recipient",
            "mixed_proven_absence",
            "unproven",
        }
        for label, target_receipt in targets.items():
            if not isinstance(target_receipt, Mapping):
                reject(f"{label} target receipt is not a mapping")
            if (
                target_receipt.get("authority_sha256") != authority_sha256
                or target_receipt.get("plan_sha256") != plan_sha256
                or target_receipt.get("post_puf_surface_sha256")
                != post_puf_surface_sha256
                or target_receipt.get("surface_sha256") != surface_sha256
            ):
                reject(f"{label} target receipt is not bound to canonical authority")
            authority_form = target_receipt.get("authority_form")
            if authority_form not in allowed_target_forms:
                reject(f"{label} declares invalid authority form {authority_form!r}")
            proven = target_receipt.get("proven", {})
            if not isinstance(proven, Mapping):
                reject(f"{label} proven-absence receipts are not a mapping")
            unproven = target_receipt.get("unproven", {})
            if not isinstance(unproven, Mapping):
                reject(f"{label} unproven-absence counts are not a mapping")
            proven_rows = 0
            for cell, proof_receipt in proven.items():
                if not isinstance(proof_receipt, Mapping):
                    reject(f"{label} {cell} proof receipt is not a mapping")
                direction = direction_by_label.get(label)
                if direction is None:
                    reject(
                        f"{label} has no canonical gap-fill direction and cannot "
                        "carry a proven-absence receipt"
                    )
                cell_channel, separator, _clone_role = str(cell).partition("/clone_")
                if (
                    not separator
                    or not _clone_role.isdigit()
                    or cell_channel != direction.recipient_channel
                    or proof_receipt.get("authority_form") != "origin_exact_recipient"
                    or proof_receipt.get("authority_sha256") != authority_sha256
                    or proof_receipt.get("plan_sha256") != plan_sha256
                    or proof_receipt.get("post_puf_surface_sha256")
                    != post_puf_surface_sha256
                    or proof_receipt.get("surface_sha256") != surface_sha256
                    or proof_receipt.get("declared_direction") != direction.name
                    or proof_receipt.get("declared_donor_channel")
                    != direction.donor_channel
                    or proof_receipt.get("declared_recipient_channel")
                    != direction.recipient_channel
                ):
                    reject(
                        f"{label} {cell} proof is not recipient-exact canonical authority"
                    )
                proven_rows += nonnegative_int(
                    proof_receipt,
                    "null_rows",
                    label=f"{label} {cell} proof",
                )
            unproven_rows = 0
            for cell, count in unproven.items():
                cell_channel, separator, clone_role = str(cell).partition("/clone_")
                if (
                    not separator
                    or not clone_role.isdigit()
                    or not isinstance(cell_channel, str)
                    or not cell_channel
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    reject(f"{label} unproven-absence receipt is malformed")
                unproven_rows += count
            status = target_receipt.get("status")
            null_rows = target_receipt.get("null_rows")
            if passed:
                if status not in {"complete", "proven_absent"}:
                    reject(f"{label} passing completeness status is {status!r}")
                if (
                    isinstance(null_rows, bool)
                    or not isinstance(null_rows, int)
                    or null_rows < 0
                ):
                    reject(f"{label} passing null_rows is not a non-negative integer")
                if unproven_rows:
                    reject(f"{label} passing receipt carries unproven nulls")
                invalid_rows = nonnegative_int(
                    target_receipt,
                    "invalid_rows",
                    label=label,
                )
                if invalid_rows:
                    reject(f"{label} passing receipt carries invalid values")
                if status == "complete" and (
                    null_rows != 0
                    or proven_rows != 0
                    or authority_form != "observed_complete"
                ):
                    reject(f"{label} complete receipt has contradictory null authority")
                if status == "proven_absent" and (
                    null_rows <= 0
                    or proven_rows != null_rows
                    or authority_form != "origin_exact_recipient"
                ):
                    reject(f"{label} proven-absence count arithmetic is inconsistent")

        rent_label = "person/housing/pre_subsidy_rent"
        rent_target = targets[rent_label]
        validate_rent_structure = passed or rent_target.get("status") not in {
            "missing",
            "missing_entity",
        }
        if validate_rent_structure:
            rent_rows, rent_by_role = validate_structural_absence_receipt(
                rent_target.get("recipient_absence_authority"),
                label=rent_label,
                battery=False,
            )
        else:
            rent_rows, rent_by_role = 0, {}
        if passed:
            rent_null_rows = nonnegative_int(
                rent_target,
                "null_rows",
                label=rent_label,
            )
            if rent_null_rows != rent_rows:
                reject(f"{rent_label} null count differs from structural authority")
            rent_proven = rent_target.get("proven", {})
            if not isinstance(rent_proven, Mapping):
                reject(f"{rent_label} proven-absence receipts are not a mapping")
            if set(rent_proven) != set(rent_by_role):
                reject(f"{rent_label} proven roles differ from structural authority")
            direction = direction_by_label[rent_label]
            expected_proof_fields = {
                "null_rows",
                "reason",
                "authority_form",
                "authority_sha256",
                "plan_sha256",
                "post_puf_surface_sha256",
                "surface_sha256",
                "declared_direction",
                "declared_donor_channel",
                "declared_recipient_channel",
                "structural_absence_rule_id",
                "structural_absence_selection",
            }
            for cell, count in rent_by_role.items():
                proof = rent_proven[cell]
                if (
                    not isinstance(proof, Mapping)
                    or set(proof) != expected_proof_fields
                    or proof.get("null_rows") != count
                    or proof.get("reason") != _ACS_GQ_RENT_ABSENCE_REASON
                    or proof.get("structural_absence_rule_id")
                    != _ACS_GQ_RENT_ABSENCE_RULE_ID
                    or proof.get("structural_absence_selection")
                    != _ACS_GQ_RENT_ABSENCE_SELECTION
                    or proof.get("declared_direction") != direction.name
                ):
                    reject(f"{rent_label} {cell} structural proof is not canonical")
        return

    if gate_name == _BATTERY_GATE_NAME:
        expected_labels = {
            *(_battery_target_label(target) for target in expected_keys),
            *(
                _joint_battery_target_label(target)
                for target in _canonical_joint_registry
            ),
        }
        expected_plan = {
            "plan_id": "stacked_gap_fill_plan",
            "version": authority["version"],
            "sha256": plan_sha256,
        }
        comparisons = details.get("comparisons")
        if (
            details.get("declared_target_count") != len(expected_keys)
            or details.get("registered_target_count") != len(expected_keys)
            or details.get("registered_joint_target_count")
            != len(_canonical_joint_registry)
            or details.get("missing_declared_targets") != []
            or details.get("extra_registered_targets") != []
        ):
            reject(
                "canonical battery coverage receipt must bind all "
                f"{len(expected_keys)} targets"
            )
        if details.get("declared_plan") != expected_plan:
            reject("canonical battery plan receipt mismatch")
        if details.get("support_profile") != authority["components"]["support_profile"]:
            reject("canonical battery support-profile receipt mismatch")
        if not isinstance(comparisons, Mapping) or set(comparisons) != expected_labels:
            reject("canonical battery comparison surface mismatch")
        for target, metric in _canonical_registry.items():
            label = _battery_target_label(target)
            comparison = comparisons[label]
            if (
                not isinstance(comparison, Mapping)
                or comparison.get("metric") != metric
            ):
                reject(f"{label} comparison must use canonical metric {metric!r}")
        for target, metric in _canonical_joint_registry.items():
            label = _joint_battery_target_label(target)
            comparison = comparisons[label]
            if (
                not isinstance(comparison, Mapping)
                or comparison.get("metric") != metric
            ):
                reject(f"{label} comparison must use canonical metric {metric!r}")
        if passed:
            allowed_statuses = {"tested", "insufficient_support"}
            statuses = {
                label: comparison.get("status")
                for label, comparison in comparisons.items()
                if isinstance(comparison, Mapping)
            }
            invalid_statuses = {
                label: status
                for label, status in statuses.items()
                if status not in allowed_statuses
            }
            if invalid_statuses:
                reject(
                    "passing battery carries failing comparison statuses "
                    f"{invalid_statuses}"
                )
            tested_labels = {
                label for label, status in statuses.items() if status == "tested"
            }
            untestable_labels = sorted(
                label
                for label, status in statuses.items()
                if status == "insufficient_support"
            )
            if details.get("tested_comparisons") != len(tested_labels):
                reject("passing battery tested-comparison count is inconsistent")
            if details.get("untestable_comparisons") != untestable_labels:
                reject("passing battery untestable-comparison list is inconsistent")

        rent_label = "person/housing/pre_subsidy_rent[clone_0]"
        rent_comparison = comparisons[rent_label]
        if not isinstance(rent_comparison, Mapping):
            reject(f"{rent_label} comparison is not a mapping")
        if passed or rent_comparison.get("status") != "missing_column":
            validate_structural_absence_receipt(
                rent_comparison.get("recipient_absence_authority"),
                label=rent_label,
                battery=True,
            )
        return

    reject("unknown stacked authority gate")


_canonical_surface_keys = _surface_target_keys(
    _CANONICAL_STACKED_DECLARED_SURFACE_ANCHOR
)
_canonical_early_transfer_keys = set(
    _surface_target_keys(_CANONICAL_STACKED_GAP_FILL_SURFACE_ANCHOR)
)
_canonical_late_transfer_keys = set(
    _surface_target_keys(_CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE_ANCHOR)
)
_canonical_late_puf_producer_keys = set(
    _surface_target_keys(_CANONICAL_STACKED_POST_PUF_PUF_PRODUCER_SURFACE_ANCHOR)
)
_canonical_late_source_producer_keys = set(
    _surface_target_keys(_CANONICAL_STACKED_POST_PUF_SOURCE_PRODUCER_SURFACE_ANCHOR)
)
_canonical_full_transfer_keys = set(
    _surface_target_keys(_freeze_target_families(pool_transfer_target_families()))
)
if (
    len(_canonical_surface_keys) != 131
    or len(set(_canonical_surface_keys)) != 131
    or len(_canonical_early_transfer_keys) != 48
    or len(_canonical_late_transfer_keys) != 70
    or len(_canonical_late_puf_producer_keys) != 43
    or len(_canonical_late_source_producer_keys) != 29
    or len(_canonical_late_puf_producer_keys & _canonical_late_source_producer_keys)
    != 2
    or _canonical_late_puf_producer_keys | _canonical_late_source_producer_keys
    != _canonical_late_transfer_keys
    or _canonical_early_transfer_keys & _canonical_late_transfer_keys
    or _canonical_early_transfer_keys | _canonical_late_transfer_keys
    != _canonical_full_transfer_keys
    or len(_canonical_full_transfer_keys) != 118
    or set(_plan_target_keys(_CANONICAL_STACKED_GAP_FILL_PLAN_ANCHOR))
    != _canonical_early_transfer_keys
    or not _canonical_full_transfer_keys.issubset(_canonical_surface_keys)
    or set(_CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY_ANCHOR)
    != set(_canonical_surface_keys)
    or len(_CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY_ANCHOR) != 1
    or any(
        (entity, family, column, clone_index) not in _canonical_surface_keys
        or _CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY_ANCHOR[
            (entity, family, column, clone_index)
        ]
        != "categorical_tvd"
        for entity, family, columns, clone_index in (
            _CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY_ANCHOR
        )
        for column in columns
    )
):
    raise RuntimeError(
        "Canonical stacked authority must partition the exact 118-target "
        "transfer surface into 48 early gap-fill and 70 post-PUF targets "
        "inside an exact 131-target terminal surface and metric registry; "
        "the late surface must be exactly covered by 43 PUF-clone and 29 "
        "ASEC-source producer targets with their declared two-target overlap."
    )


def gap_fill_stacked_spine(
    frame: Frame,
    *,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_banks: Mapping[str, AcsTransferTargetBank] | None = None,
) -> GapFillResult:
    """Run the canonical stacked gap-fill plan with no caller authority."""

    return _gap_fill_stacked_spine_evaluate(
        frame,
        authority=_production_stacked_authority(),
        production=True,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
        target_banks=target_banks,
    )


def _gap_fill_stacked_spine_with_test_authority(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_banks: Mapping[str, AcsTransferTargetBank] | None = None,
) -> GapFillResult:
    """Explicit non-production seam for fixture-sized authority surfaces."""

    _validate_test_authority(authority, boundary="stacked gap-fill test seam")
    return _gap_fill_stacked_spine_evaluate(
        frame,
        authority=authority,
        production=False,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
        target_banks=target_banks,
    )


def _gap_fill_stacked_spine_evaluate(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    production: bool,
    seed: int,
    n_estimators: int,
    max_targets_per_fit: int,
    target_banks: Mapping[str, AcsTransferTargetBank] | None,
) -> GapFillResult:
    """Gap-fill survey-specific fields cross-origin on the stacked spine.

    Runs before any clone operator: every row still carries clone index zero,
    so filled values are cloned into the PUF arm afterwards and the single
    PUF pass conditions on observed predictors for every origin.

    Per direction, in order:

    1. **Activation authority** — declared, then verified: every target
       column must exist, the donor origin's rows must observe it completely,
       and every null cell must lie on the declared recipient origin.  A
       null anywhere else is a named terminal failure, so absence can never
       silently reroute or skip a family (microcosm#578 audit item 2).
    2. **Authoritative donors** — the donor frame passed to the spine-blind
       transfer is this owner's projection of the donor origin's native rows
       (audit item 3); ``donor_channel=None`` marks the deliberate
       whole-donor fit of that projection.
    3. **Banked transfer** — the reviewed #608 target-at-a-time banking
       machinery is reused unchanged via ``target_banks[direction.name]``.
    4. **Post-verification** — donor-origin cells must be byte-identical
       before and after, and no null may remain on authorized rows beyond
       the transfer's receipted unmodeled rows.

    Returns a :class:`GapFillResult` whose receipt records, per direction and
    target, the authorized-null, imputed, unmodeled, and residual-null
    counts alongside the transfer's fit provenance.
    """

    authority_receipt = _authority_receipt(authority)
    authority_failures = _authority_validation_failures(
        authority,
        production=production,
    )
    if authority_failures:
        raise ValueError(
            "Stacked gap-fill authority validation failed:\n  "
            + "\n  ".join(authority_failures)
        )
    if production:
        _validate_production_authority_receipt(
            authority_receipt,
            boundary="stacked gap-fill entry",
        )
    validate_stacked_spine_frame(frame, boundary="stacked gap-fill entry")
    directions = authority.gap_fill_plan
    if not directions:
        raise ValueError("Stacked gap-fill requires at least one direction.")
    names = [direction.name for direction in directions]
    if len(set(names)) != len(names):
        raise ValueError(f"Stacked gap-fill directions repeat names: {names}.")
    if target_banks is not None:
        unknown_banks = sorted(set(target_banks) - set(names))
        if unknown_banks:
            raise ValueError(
                f"target_banks name unknown gap-fill direction(s): {unknown_banks}."
            )

    person_clone = frame.table("person")[support_clone_index_column("person")]
    if not person_clone.eq(0).all():
        raise ValueError(
            "Stacked gap-fill must run before clone operators; found nonzero "
            "person support clone indices."
        )

    current = frame
    receipts: dict[str, object] = {}
    transfer_results: dict[str, AcsTransferResult] = {}
    for direction in directions:
        pre_counts = _verify_gap_fill_activation_authority(
            current,
            direction=direction,
        )
        donor = _origin_projection(current, channel=direction.donor_channel)
        donor_snapshot = {
            entity: _direction_targets_snapshot(
                current,
                entity=entity,
                targets=targets,
                channel=direction.donor_channel,
            )
            for entity, targets in _direction_entity_targets(direction).items()
        }
        result = transfer_acs_inputs(
            current,
            donor,
            target_families=direction.target_families,
            donor_channel=None,
            seed=seed,
            n_estimators=n_estimators,
            max_targets_per_fit=max_targets_per_fit,
            target_bank=(target_banks or {}).get(direction.name),
        )
        transfer_results[direction.name] = result
        current = result.frame
        receipts[direction.name] = _verify_gap_fill_outcome(
            current,
            direction=direction,
            pre_counts=pre_counts,
            donor_snapshot=donor_snapshot,
            result=result,
        )

    validate_stacked_spine_frame(current, boundary="stacked gap-fill output")
    return GapFillResult(
        frame=current,
        receipt={"authority": authority_receipt, "directions": receipts},
        transfer_results=transfer_results,
    )


def _direction_entity_targets(
    direction: GapFillDirection,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for entity, families in direction.target_families.items():
        collected: list[str] = []
        for targets in families.values():
            collected.extend(targets)
        result[entity] = tuple(collected)
    return result


def _direction_absence_rule_index(
    direction: GapFillDirection,
) -> dict[tuple[str, str], GapFillAbsenceRule]:
    return {
        (rule.entity, rule.column): rule for rule in direction.recipient_absence_rules
    }


def _gap_fill_absence_rule_mask(
    frame: Frame,
    *,
    direction: GapFillDirection,
    rule: GapFillAbsenceRule,
) -> tuple[pd.Series, dict[str, object]]:
    """Resolve one structural-null rule to an exact, live row mask and receipt."""

    if (
        rule.rule_id != _ACS_GQ_RENT_ABSENCE_RULE_ID
        or rule.selection != _ACS_GQ_RENT_ABSENCE_SELECTION
        or rule.entity != "person"
        or rule.column != "pre_subsidy_rent"
        or direction.recipient_channel != ACS_STACKED_SUPPORT_CHANNEL
    ):
        raise ValueError(
            f"Gap-fill direction {direction.name!r} declares unsupported "
            f"recipient absence rule {rule.rule_id!r}."
        )

    manifest = frame.metadata.get(STACKED_SPINE_MANIFEST_KEY)
    if not isinstance(manifest, Mapping):
        raise ValueError(
            f"Gap-fill absence rule {rule.rule_id!r} requires the stacked "
            "assembly manifest."
        )
    gq_households, mask = _validated_acs_native_group_quarters_masks(
        frame,
        manifest,
        boundary=f"gap-fill absence rule {rule.rule_id!r}",
    )
    household = frame.table("household")
    person = frame.table("person")
    required_household = {
        "tenure_type",
        support_channel_column("household"),
    }
    required_person = {
        "person_household_id",
        support_channel_column("person"),
        support_clone_index_column("person"),
    }
    missing_household = sorted(required_household - set(household.columns))
    missing_person = sorted(required_person - set(person.columns))
    if missing_household or missing_person:
        raise ValueError(
            f"Gap-fill absence rule {rule.rule_id!r} cannot resolve its exact "
            "universe; "
            f"missing_household={missing_household}, missing_person={missing_person}."
        )

    nonnull_gq_tenure = gq_households & household["tenure_type"].notna()
    if nonnull_gq_tenure.any():
        raise ValueError(
            f"Gap-fill absence rule {rule.rule_id!r} found "
            f"{int(nonnull_gq_tenure.sum())} ACS group-quarters household row(s) "
            "with synthesized tenure."
        )
    person_channel = person[support_channel_column("person")].astype(str)

    clone_index = pd.to_numeric(
        person[support_clone_index_column("person")], errors="raise"
    ).astype("int64")
    by_origin_role = {
        f"{channel}/clone_{int(clone)}": int(count)
        for (channel, clone), count in (
            pd.DataFrame(
                {
                    "channel": person_channel.loc[mask],
                    "clone_index": clone_index.loc[mask],
                }
            )
            .groupby(["channel", "clone_index"], sort=True)
            .size()
            .items()
        )
    }
    return mask, {
        "rule_id": rule.rule_id,
        "selection": rule.selection,
        "reason": rule.reason,
        "status": "exact_structural_absence",
        "rows": int(mask.sum()),
        "by_origin_role": by_origin_role,
    }


def _origin_projection(frame: Frame, *, channel: str) -> Frame:
    """Project one origin's native lineages as a standalone donor frame."""

    person = frame.table("person")
    mask = (
        person[support_channel_column("person")].astype(str).eq(channel)
        & person[support_clone_index_column("person")].eq(0)
    ).to_numpy()
    if not mask.any():
        raise ValueError(
            f"Stacked spine has no native person rows for origin {channel!r}."
        )
    return frame.select(mask)


def _direction_targets_snapshot(
    frame: Frame,
    *,
    entity: str,
    targets: Sequence[str],
    channel: str,
) -> pd.DataFrame:
    table = frame.table(entity)
    mask = table[support_channel_column(entity)].astype(str).eq(channel)
    present = [target for target in targets if target in table.columns]
    return table.loc[mask, present].copy(deep=True)


def _canonical_donor_series_payload(
    series: pd.Series,
    *,
    boundary: str,
) -> tuple[object, ...]:
    """Return a canonical, byte-aware identity payload for one donor target."""

    column = "__stacked_gap_fill_donor_target__"
    canonical_table = canonicalize_table_string_dtypes(
        series.to_frame(name=column),
        boundary=boundary,
        table_name=f"donor_target[{series.name!r}]",
    )
    canonical = canonical_table[column]
    canonical.name = series.name
    values = canonical.to_numpy(copy=False)
    if (
        not pd.api.types.is_extension_array_dtype(canonical.dtype)
        and not values.dtype.hasobject
    ):
        encoding = "raw_numpy_c_order"
        value_payload = np.ascontiguousarray(values).tobytes(order="C")
    else:
        encoding = "independent_scalar_pickle_protocol_5"
        semantic_values = canonical.to_numpy(
            dtype=object,
            copy=True,
        ).tolist()
        value_payload = _semantic_scalar_sequence_payload(
            semantic_values,
            boundary=f"{boundary} values",
        )
    return (
        canonical.shape,
        _index_identity_payload(
            canonical.index,
            boundary=f"{boundary} index",
        ),
        _semantic_scalar_payload(
            canonical.name,
            boundary=f"{boundary} series name",
        ),
        (
            _qualified_type_name(canonical.dtype),
            pickle.dumps(canonical.dtype, protocol=5),
        ),
        encoding,
        value_payload,
    )


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _semantic_scalar_payload(
    value: object,
    *,
    boundary: str,
) -> tuple[str, bytes]:
    """Serialize one supported scalar without a cross-value pickle memo."""

    supported = (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or isinstance(
            value,
            (
                str,
                bytes,
                bool,
                int,
                float,
                complex,
                np.generic,
                pd.Timestamp,
                pd.Timedelta,
                pd.Period,
                pd.Interval,
            ),
        )
    )
    if not supported:
        raise TypeError(
            f"{boundary}: donor byte identity found unsupported semantic "
            f"scalar type {_qualified_type_name(value)!r}."
        )
    return (
        _qualified_type_name(value),
        pickle.dumps(value, protocol=5),
    )


def _semantic_scalar_sequence_payload(
    values: Sequence[object],
    *,
    boundary: str,
) -> tuple[tuple[str, bytes], ...]:
    payloads: list[tuple[str, bytes]] = []
    for position, value in enumerate(values):
        payload = _semantic_scalar_payload(
            value,
            boundary=f"{boundary} position {position}",
        )
        payloads.append(payload)
    return tuple(payloads)


def _index_identity_payload(
    index: pd.Index,
    *,
    boundary: str,
) -> tuple[object, ...]:
    """Return exact index authority without list-wide object serialization."""

    names = tuple(
        _semantic_scalar_payload(
            name,
            boundary=f"{boundary} name {position}",
        )
        for position, name in enumerate(index.names)
    )
    index_type = _qualified_type_name(index)
    if isinstance(index, pd.MultiIndex):
        levels = tuple(
            _index_identity_payload(
                level,
                boundary=f"{boundary} level {position}",
            )
            for position, level in enumerate(index.levels)
        )
        codes = tuple(
            (
                code.dtype.str,
                code.shape,
                np.ascontiguousarray(code).tobytes(order="C"),
            )
            for code in index.codes
        )
        return (index_type, names, "multiindex_levels_codes", levels, codes)

    dtype = index.dtype
    dtype_authority = (
        _qualified_type_name(dtype),
        pickle.dumps(dtype, protocol=5),
    )
    values = index.to_numpy(copy=False)
    if not pd.api.types.is_extension_array_dtype(dtype) and not values.dtype.hasobject:
        encoding = "raw_numpy_c_order"
        value_payload: object = (
            values.shape,
            np.ascontiguousarray(values).tobytes(order="C"),
        )
    else:
        encoding = "independent_scalar_pickle_protocol_5"
        semantic_values = index.to_numpy(dtype=object, copy=True).tolist()
        value_payload = _semantic_scalar_sequence_payload(
            semantic_values,
            boundary=f"{boundary} values",
        )
    return (index_type, names, dtype_authority, encoding, value_payload)


def _verify_gap_fill_activation_authority(
    frame: Frame,
    *,
    direction: GapFillDirection,
) -> dict[tuple[str, str], dict[str, int]]:
    """Verify declared activation authority before any modeling runs."""

    failures: list[str] = []
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for entity, families in direction.target_families.items():
        table = frame.table(entity)
        channel = table[support_channel_column(entity)].astype(str)
        recipient_rows = channel.eq(direction.recipient_channel)
        donor_rows = channel.eq(direction.donor_channel)
        recipient_count = int(recipient_rows.sum())
        donor_count = int(donor_rows.sum())
        if recipient_count == 0:
            failures.append(
                f"{direction.name}/{entity}: declared recipient channel "
                f"{direction.recipient_channel!r} has no live rows."
            )
        if donor_count == 0:
            failures.append(
                f"{direction.name}/{entity}: declared donor channel "
                f"{direction.donor_channel!r} has no live rows."
            )
        for family, targets in families.items():
            for target in targets:
                label = f"{direction.name}/{entity}/{family}/{target}"
                if target not in table.columns:
                    failures.append(
                        f"{label}: declared gap-fill target column is absent "
                        "from the stacked spine."
                    )
                    continue
                null_mask = table[target].isna()
                donor_nulls = int((null_mask & donor_rows).sum())
                unauthorized_nulls = int(
                    (null_mask & ~recipient_rows & ~donor_rows).sum()
                )
                if donor_nulls:
                    failures.append(
                        f"{label}: donor origin {direction.donor_channel!r} "
                        f"has {donor_nulls} null cell(s); donors must observe "
                        "every declared target."
                    )
                if unauthorized_nulls:
                    failures.append(
                        f"{label}: {unauthorized_nulls} null cell(s) lie "
                        "outside the declared recipient origin "
                        f"{direction.recipient_channel!r}."
                    )
                counts[(entity, target)] = {
                    "authorized_null_rows": int((null_mask & recipient_rows).sum()),
                    "recipient_rows": int(recipient_rows.sum()),
                    "donor_rows": int(donor_rows.sum()),
                }
    if failures:
        raise ValueError(
            "Stacked gap-fill activation authority failed:\n  " + "\n  ".join(failures)
        )
    return counts


def _verify_gap_fill_outcome(
    frame: Frame,
    *,
    direction: GapFillDirection,
    pre_counts: Mapping[tuple[str, str], Mapping[str, int]],
    donor_snapshot: Mapping[str, pd.DataFrame],
    result: AcsTransferResult,
) -> dict[str, object]:
    """Verify donor invariance and residual nulls; build the direction receipt."""

    failures: list[str] = []
    imputed_by_target = {
        (record.entity, record.column): record for record in result.imputed_inputs
    }
    target_receipts: dict[str, dict[str, object]] = {}
    absence_rules = _direction_absence_rule_index(direction)
    for entity, families in direction.target_families.items():
        table = frame.table(entity)
        channel = table[support_channel_column(entity)].astype(str)
        recipient_rows = channel.eq(direction.recipient_channel)
        donor_after = {
            entity_name: _direction_targets_snapshot(
                frame,
                entity=entity_name,
                targets=targets,
                channel=direction.donor_channel,
            )
            for entity_name, targets in _direction_entity_targets(direction).items()
        }[entity]
        for family, targets in families.items():
            for target in targets:
                label = f"{direction.name}/{entity}/{family}/{target}"
                before = donor_snapshot[entity].get(target)
                after = donor_after.get(target)
                if (
                    before is None
                    or after is None
                    or _canonical_donor_series_payload(
                        before,
                        boundary=f"{label} donor identity before transfer",
                    )
                    != _canonical_donor_series_payload(
                        after,
                        boundary=f"{label} donor identity after transfer",
                    )
                ):
                    failures.append(
                        f"{label}: donor byte identity failed for origin "
                        f"{direction.donor_channel!r}; canonical donor payload "
                        "changed during gap-fill transfer."
                    )
                null_mask = table[target].isna()
                residual_mask = null_mask & recipient_rows
                residual_nulls = int(residual_mask.sum())
                outside_nulls = int((null_mask & ~recipient_rows).sum())
                if outside_nulls:
                    failures.append(
                        f"{label}: {outside_nulls} null cell(s) appeared "
                        "outside the declared recipient origin during the "
                        "transfer."
                    )
                record = imputed_by_target.get((entity, target))
                imputed = record.imputed_recipient_rows if record else 0
                unmodeled = record.unmodeled_recipient_rows if record else 0
                pre = pre_counts[(entity, target)]
                authorized = pre["authorized_null_rows"]
                if residual_nulls != unmodeled:
                    failures.append(
                        f"{label}: residual-null equation failed: "
                        f"residual_null_rows={residual_nulls} != "
                        f"unmodeled_rows={unmodeled}."
                    )
                if authorized != imputed + unmodeled:
                    failures.append(
                        f"{label}: activation accounting equation failed: "
                        f"authorized_null_rows={authorized} != "
                        f"imputed_rows={imputed} + unmodeled_rows={unmodeled}."
                    )
                target_receipt: dict[str, object] = {
                    "authorized_null_rows": authorized,
                    "imputed_rows": imputed,
                    "unmodeled_rows": unmodeled,
                    "residual_null_rows": residual_nulls,
                }
                rule = absence_rules.get((entity, target))
                if rule is not None:
                    expected_absence, absence_receipt = _gap_fill_absence_rule_mask(
                        frame,
                        direction=direction,
                        rule=rule,
                    )
                    unexpected = int((residual_mask & ~expected_absence).sum())
                    synthesized = int((expected_absence & ~residual_mask).sum())
                    if unexpected or synthesized:
                        failures.append(
                            f"{label}: exact structural-absence equation failed; "
                            f"unexpected_null_rows={unexpected}, "
                            f"structural_rows_filled={synthesized}."
                        )
                    target_receipt["recipient_absence_authority"] = {
                        **absence_receipt,
                        "unexpected_null_rows": unexpected,
                        "structural_rows_filled": synthesized,
                    }
                elif unmodeled or residual_nulls:
                    failures.append(
                        f"{label}: undeclared gap-fill residual is forbidden; "
                        f"unmodeled_rows={unmodeled}, "
                        f"residual_null_rows={residual_nulls}. Every downstream "
                        "consumer requires this early target complete."
                    )
                target_receipts[f"{entity}/{family}/{target}"] = target_receipt
    if failures:
        raise ValueError(
            "Stacked gap-fill outcome verification failed:\n  " + "\n  ".join(failures)
        )
    return {
        "recipient_channel": direction.recipient_channel,
        "donor_channel": direction.donor_channel,
        "donor_selection": "owner_projection_of_native_donor_rows",
        "resolved_donor_channel": result.resolved_donor_channel,
        "targets": target_receipts,
        "deferred_inputs": list(result.deferred_inputs),
        "fit_records": [
            {"fit_name": record.fit_name, "weight_kind": record.weight_kind}
            for record in result.fit_records
        ],
    }


# ---------------------------------------------------------------------------
# Post-PUF transfer of outputs that do not exist at the early gap-fill stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackedPostPufTransferResult:
    """The completed stacked frame plus late-transfer provenance."""

    frame: Frame
    receipt: Mapping[str, object]
    transfer_result: AcsTransferResult


@dataclass(frozen=True)
class StackedLateProducerResult:
    """A fully executed late-producer DAG and its aggregate provenance."""

    frame: Frame
    receipt: Mapping[str, object]
    primary_puf_result: StackedPufPassResult
    source_completion_receipt: Mapping[str, object]
    transfer_result: AcsTransferResult
    transition_authority_sha256: str


def _producer_role_surface_for_group(
    group_surface: TargetFamilies,
    producer_surface: TargetFamilies,
) -> TargetFamilies:
    """Project canonical producer roles onto one bounded transfer family."""

    producer_targets = {
        (entity, target)
        for entity, families in producer_surface.items()
        for targets in families.values()
        for target in targets
    }
    return {
        entity: {
            family: tuple(
                target for target in targets if (entity, target) in producer_targets
            )
        }
        for entity, families in group_surface.items()
        for family, targets in families.items()
        if any((entity, target) in producer_targets for target in targets)
    }


def transfer_stacked_post_puf_group(
    frame: Frame,
    *,
    group_name: str,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_bank: AcsTransferTargetBank | None = None,
    execution_contract: Mapping[str, object] | None = None,
) -> StackedPostPufTransferResult:
    """Execute one canonical bounded late-transfer producer."""

    groups = {group.name: group for group in CANONICAL_US_LATE_TRANSFER_GROUPS}
    if group_name not in groups:
        raise ValueError(
            f"Unknown canonical US late-transfer producer {group_name!r}; "
            f"expected one of {sorted(groups)}."
        )
    group = groups[group_name]
    authority = _production_stacked_authority()
    result = _transfer_stacked_post_puf_inputs_evaluate(
        frame,
        authority=authority,
        production=True,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
        target_bank=target_bank,
        target_families=group.target_families,
        derive_schedule_d=False,
        execution_contract=execution_contract,
    )
    return StackedPostPufTransferResult(
        frame=result.frame,
        receipt={
            **dict(result.receipt),
            "producer": group.name,
            "entity": group.entity,
            "family": group.family,
            "ordered_targets": list(group.targets),
        },
        transfer_result=result.transfer_result,
    )


def transfer_stacked_post_puf_inputs(
    frame: Frame,
    *,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_bank: AcsTransferTargetBank | None = None,
) -> StackedPostPufTransferResult:
    """Transfer canonical late-produced inputs after source completion."""

    return _transfer_stacked_post_puf_inputs_evaluate(
        frame,
        authority=_production_stacked_authority(),
        production=True,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
        target_bank=target_bank,
        target_families=None,
        derive_schedule_d=True,
        execution_contract=None,
    )


def _transfer_stacked_post_puf_inputs_with_test_authority(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_bank: AcsTransferTargetBank | None = None,
) -> StackedPostPufTransferResult:
    """Explicit non-production seam for fixture-sized late surfaces."""

    _validate_test_authority(authority, boundary="post-PUF transfer test seam")
    return _transfer_stacked_post_puf_inputs_evaluate(
        frame,
        authority=authority,
        production=False,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
        target_bank=target_bank,
        target_families=None,
        derive_schedule_d=True,
        execution_contract=None,
    )


def _transfer_stacked_post_puf_inputs_evaluate(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    production: bool,
    seed: int,
    n_estimators: int,
    max_targets_per_fit: int,
    target_bank: AcsTransferTargetBank | None,
    target_families: TargetFamilies | None,
    derive_schedule_d: bool,
    execution_contract: Mapping[str, object] | None,
) -> StackedPostPufTransferResult:
    """Run the late transfer from the one role carrying every declared target."""

    authority_receipt = _authority_receipt(authority)
    authority_failures = _authority_validation_failures(
        authority,
        production=production,
    )
    if authority_failures:
        raise ValueError(
            "Stacked post-PUF transfer authority validation failed:\n  "
            + "\n  ".join(authority_failures)
        )
    if production:
        _validate_production_authority_receipt(
            authority_receipt,
            boundary="stacked post-PUF transfer entry",
        )
    validate_stacked_spine_frame(frame, boundary="stacked post-PUF transfer entry")
    validate_puf_clone_attachment(
        frame,
        boundary="stacked post-PUF transfer attachment",
    )
    surface = (
        authority.post_puf_transfer_surface
        if target_families is None
        else target_families
    )
    if not _surface_target_keys(surface):
        raise ValueError("Stacked post-PUF transfer requires at least one target.")
    puf_producer_surface = (
        authority.post_puf_puf_producer_surface
        if target_families is None
        else _producer_role_surface_for_group(
            surface,
            authority.post_puf_puf_producer_surface,
        )
    )
    source_producer_surface = (
        authority.post_puf_source_producer_surface
        if target_families is None
        else _producer_role_surface_for_group(
            surface,
            authority.post_puf_source_producer_surface,
        )
    )

    pre_counts = _verify_post_puf_transfer_activation_authority(
        frame,
        target_families=surface,
        puf_producer_families=puf_producer_surface,
        source_producer_families=source_producer_surface,
    )
    donor = _post_puf_donor_projection(frame)
    puf_producer_keys = set(_surface_target_keys(puf_producer_surface))
    source_producer_keys = set(_surface_target_keys(source_producer_surface))
    producer_snapshot = {
        (entity, target): _post_puf_producer_snapshot(
            frame,
            entity=entity,
            target=target,
            puf_produced=(entity, family, target, 0) in puf_producer_keys,
            source_produced=(entity, family, target, 0) in source_producer_keys,
        )
        for entity, families in surface.items()
        for family, targets in families.items()
        for target in targets
    }
    transfer = transfer_acs_inputs(
        frame,
        donor,
        target_families=surface,
        donor_spine=ASEC_PUF_DONOR_SPINE,
        donor_channel=None,
        seed=seed,
        n_estimators=n_estimators,
        max_targets_per_fit=max_targets_per_fit,
        target_bank=target_bank,
        derive_schedule_d=derive_schedule_d,
        execution_contract=execution_contract,
    )
    target_receipts = _verify_post_puf_transfer_outcome(
        transfer.frame,
        target_families=surface,
        puf_producer_families=puf_producer_surface,
        source_producer_families=source_producer_surface,
        pre_counts=pre_counts,
        producer_snapshot=producer_snapshot,
        result=transfer,
    )
    validate_stacked_spine_frame(
        transfer.frame,
        boundary="stacked post-PUF transfer output",
    )
    return StackedPostPufTransferResult(
        frame=transfer.frame,
        receipt={
            "authority": authority_receipt,
            "donor_selection": "owner_projection_of_asec_origin_clone_1",
            "donor_channel": BASE_ASEC_SUPPORT_CHANNEL,
            "donor_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
            "recipient_selection": (
                "target_specific_complement_of_declared_producer_rows"
            ),
            "resolved_donor_channel": transfer.resolved_donor_channel,
            "targets": target_receipts,
            "fit_records": [
                {"fit_name": record.fit_name, "weight_kind": record.weight_kind}
                for record in transfer.fit_records
            ],
        },
        transfer_result=transfer,
    )


def _post_puf_role_mask(frame: Frame, *, entity: str) -> pd.Series:
    table = frame.table(entity)
    return table[support_channel_column(entity)].astype(str).eq(
        BASE_ASEC_SUPPORT_CHANNEL
    ) & pd.to_numeric(
        table[support_clone_index_column(entity)],
        errors="raise",
    ).eq(PUF_TAX_DETAIL_CLONE_INDEX)


def _post_puf_donor_projection(frame: Frame) -> Frame:
    person_mask = _post_puf_role_mask(frame, entity=frame.schema.person_entity)
    if not person_mask.any():
        raise ValueError(
            "Stacked post-PUF transfer has no ASEC-origin clone-1 donor rows."
        )
    return frame.select(person_mask.to_numpy(dtype=bool))


def _verify_post_puf_cross_grain_clone_provenance(frame: Frame) -> None:
    """Require each person to share the clone role of every parent entity."""

    person_entity = frame.schema.person_entity
    person = frame.table(person_entity)
    person_clone = pd.to_numeric(
        person[support_clone_index_column(person_entity)],
        errors="raise",
    )
    failures: list[str] = []
    for group in frame.schema.group_entities:
        membership_column = frame.schema.membership_column(group)
        group_id_column = frame.schema.entity_id_column(group)
        group_table = frame.table(group)
        group_clone = pd.to_numeric(
            group_table.set_index(group_id_column)[support_clone_index_column(group)],
            errors="raise",
        )
        expected = person[membership_column].map(group_clone)
        mismatch = expected.isna() | expected.ne(person_clone)
        if mismatch.any():
            failures.append(
                f"{int(mismatch.sum())} person/{group} link(s) disagree on "
                "support clone index"
            )
    if failures:
        raise ValueError(
            "Stacked post-PUF transfer cross-grain clone provenance failed:\n  "
            + "\n  ".join(failures)
        )


def _post_puf_producer_mask(
    frame: Frame,
    *,
    entity: str,
    puf_produced: bool,
    source_produced: bool,
) -> pd.Series:
    table = frame.table(entity)
    producer_rows = pd.Series(False, index=table.index, dtype=bool)
    if puf_produced:
        producer_rows |= pd.to_numeric(
            table[support_clone_index_column(entity)],
            errors="raise",
        ).gt(0)
    if source_produced:
        producer_rows |= (
            table[support_channel_column(entity)]
            .astype(str)
            .eq(BASE_ASEC_SUPPORT_CHANNEL)
        )
    return producer_rows


def _post_puf_producer_snapshot(
    frame: Frame,
    *,
    entity: str,
    target: str,
    puf_produced: bool,
    source_produced: bool,
) -> pd.Series:
    table = frame.table(entity)
    producer_rows = _post_puf_producer_mask(
        frame,
        entity=entity,
        puf_produced=puf_produced,
        source_produced=source_produced,
    )
    return table.loc[producer_rows, target].copy(deep=True)


def _verify_post_puf_transfer_activation_authority(
    frame: Frame,
    *,
    target_families: TargetFamilies,
    puf_producer_families: TargetFamilies,
    source_producer_families: TargetFamilies,
) -> dict[tuple[str, str], dict[str, int]]:
    """Require complete declared producers before authorizing recipient nulls."""

    failures: list[str] = []
    counts: dict[tuple[str, str], dict[str, int]] = {}
    _verify_post_puf_cross_grain_clone_provenance(frame)
    puf_producer_keys = set(_surface_target_keys(puf_producer_families))
    source_producer_keys = set(_surface_target_keys(source_producer_families))
    for entity, families in target_families.items():
        table = frame.table(entity)
        donor_rows = _post_puf_role_mask(frame, entity=entity)
        donor_count = int(donor_rows.sum())
        if donor_count == 0:
            failures.append(
                f"post_puf_transfer/{entity}: declared ASEC clone-1 donor role "
                "has no live rows."
            )
        for family, targets in families.items():
            for target in targets:
                label = f"post_puf_transfer/{entity}/{family}/{target}"
                key = (entity, family, target, 0)
                puf_produced = key in puf_producer_keys
                source_produced = key in source_producer_keys
                producer_rows = _post_puf_producer_mask(
                    frame,
                    entity=entity,
                    puf_produced=puf_produced,
                    source_produced=source_produced,
                )
                recipient_rows = ~producer_rows
                if target not in table.columns:
                    failures.append(
                        f"{label}: declared post-PUF transfer target column is "
                        "absent from the stacked spine."
                    )
                    continue
                null_mask = table[target].isna()
                producer_nulls = int((null_mask & producer_rows).sum())
                if producer_nulls:
                    roles = ", ".join(
                        role
                        for role, active in (
                            ("PUF clone", puf_produced),
                            ("ASEC source", source_produced),
                        )
                        if active
                    )
                    failures.append(
                        f"{label}: declared {roles} producer role(s) have "
                        f"{producer_nulls} null cell(s); upstream producers must "
                        "observe every producer-owned target."
                    )
                counts[(entity, target)] = {
                    "authorized_null_rows": int((null_mask & recipient_rows).sum()),
                    "recipient_rows": int(recipient_rows.sum()),
                    "producer_rows": int(producer_rows.sum()),
                    "donor_rows": donor_count,
                }
    if failures:
        raise ValueError(
            "Stacked post-PUF transfer activation authority failed:\n  "
            + "\n  ".join(failures)
        )
    return counts


def _verify_post_puf_transfer_outcome(
    frame: Frame,
    *,
    target_families: TargetFamilies,
    puf_producer_families: TargetFamilies,
    source_producer_families: TargetFamilies,
    pre_counts: Mapping[tuple[str, str], Mapping[str, int]],
    producer_snapshot: Mapping[tuple[str, str], pd.Series],
    result: AcsTransferResult,
) -> dict[str, dict[str, object]]:
    """Prove producers were preserved and every authorized null was filled."""

    failures: list[str] = []
    imputed_by_target = {
        (record.entity, record.column): record for record in result.imputed_inputs
    }
    target_receipts: dict[str, dict[str, object]] = {}
    puf_producer_keys = set(_surface_target_keys(puf_producer_families))
    source_producer_keys = set(_surface_target_keys(source_producer_families))
    for entity, families in target_families.items():
        table = frame.table(entity)
        for family, family_targets in families.items():
            for target in family_targets:
                label = f"post_puf_transfer/{entity}/{family}/{target}"
                key = (entity, family, target, 0)
                puf_produced = key in puf_producer_keys
                source_produced = key in source_producer_keys
                producer_rows = _post_puf_producer_mask(
                    frame,
                    entity=entity,
                    puf_produced=puf_produced,
                    source_produced=source_produced,
                )
                recipient_rows = ~producer_rows
                before = producer_snapshot.get((entity, target))
                after = table.loc[producer_rows, target].copy(deep=True)
                if before is None or _canonical_donor_series_payload(
                    before,
                    boundary=f"{label} producer identity before transfer",
                ) != _canonical_donor_series_payload(
                    after,
                    boundary=f"{label} producer identity after transfer",
                ):
                    failures.append(
                        f"{label}: producer byte identity failed; a declared "
                        "producer payload changed during transfer."
                    )
                null_mask = table[target].isna()
                producer_nulls = int((null_mask & producer_rows).sum())
                recipient_nulls = int((null_mask & recipient_rows).sum())
                residual_nulls = int(null_mask.sum())
                record = imputed_by_target.get((entity, target))
                imputed = record.imputed_recipient_rows if record else 0
                unmodeled = record.unmodeled_recipient_rows if record else 0
                authorized = pre_counts[(entity, target)]["authorized_null_rows"]
                if producer_nulls:
                    failures.append(
                        f"{label}: {producer_nulls} producer null cell(s) appeared "
                        "during transfer."
                    )
                if recipient_nulls != unmodeled:
                    failures.append(
                        f"{label}: residual-null equation failed: "
                        f"recipient_null_rows={recipient_nulls} != "
                        f"unmodeled_rows={unmodeled}."
                    )
                if authorized != imputed + unmodeled:
                    failures.append(
                        f"{label}: activation accounting equation failed: "
                        f"authorized_null_rows={authorized} != "
                        f"imputed_rows={imputed} + unmodeled_rows={unmodeled}."
                    )
                if unmodeled or residual_nulls:
                    failures.append(
                        f"{label}: declared post-PUF transfer left "
                        f"unmodeled_rows={unmodeled}, "
                        f"residual_null_rows={residual_nulls}; zero are allowed."
                    )
                target_receipts[f"{entity}/{family}/{target}"] = {
                    "producer_roles": [
                        role
                        for role, active in (
                            ("puf_clone", puf_produced),
                            ("asec_source", source_produced),
                        )
                        if active
                    ],
                    "producer_rows": pre_counts[(entity, target)]["producer_rows"],
                    "authorized_null_rows": authorized,
                    "imputed_rows": imputed,
                    "unmodeled_rows": unmodeled,
                    "residual_null_rows": residual_nulls,
                }
    if failures:
        raise ValueError(
            "Stacked post-PUF transfer outcome verification failed:\n  "
            + "\n  ".join(failures)
        )
    return target_receipts


def _late_required_scope_mask(
    frame: Frame,
    *,
    entity: str,
    required_scope: str,
) -> pd.Series:
    """Resolve one declared late-stage row scope without inferring absence."""

    table = frame.table(entity)
    if required_scope == "whole_pool":
        return pd.Series(True, index=table.index, dtype=bool)
    if required_scope == "asec_source":
        return (
            table[support_channel_column(entity)]
            .astype(str)
            .eq(BASE_ASEC_SUPPORT_CHANNEL)
        )
    if required_scope == "acs_source":
        return (
            table[support_channel_column(entity)]
            .astype(str)
            .eq(ACS_STACKED_SUPPORT_CHANNEL)
        )
    if required_scope == "puf_clone":
        return pd.to_numeric(
            table[support_clone_index_column(entity)],
            errors="raise",
        ).gt(0)
    raise ValueError(f"Unknown US late-producer scope {required_scope!r}.")


def _late_input_readiness_rows(
    frame: Frame,
    contract: ProducerContract,
    *,
    available_input_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[ProducerInput, int], dict[ProducerInput, int]]:
    """Count missing rows and invalid values as distinct readiness states."""

    available = (
        {} if available_input_receipts is None else dict(available_input_receipts)
    )
    unfilled: dict[ProducerInput, int] = {}
    invalid: dict[ProducerInput, int] = {}
    for requirement in contract.inputs:
        column_states = {
            input_column: _late_input_column_readiness_rows(
                frame,
                input_column=input_column,
                required_scope=requirement.required_scope,
                producer_name=contract.name,
                available_input_receipts=available,
            )
            for alternative in requirement.alternatives
            for input_column in alternative
        }
        alternative_missing_counts = [
            sum(column_states[input_column][0] for input_column in alternative)
            for alternative in requirement.alternatives
        ]
        # Invalid finite-numeric values never become absence merely because a
        # different spelling is absent.  Callbacks select alternatives by
        # physical availability, so every present declared numeric column must
        # be valid before the callback may inspect it.
        unfilled[requirement] = min(alternative_missing_counts)
        invalid[requirement] = sum(
            column_states[input_column][1] for input_column in column_states
        )
    return unfilled, invalid


def _late_unfilled_input_rows(
    frame: Frame,
    contract: ProducerContract,
    *,
    available_input_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[ProducerInput, int]:
    """Compatibility projection of the distinct late-input readiness state."""

    unfilled, _invalid = _late_input_readiness_rows(
        frame,
        contract,
        available_input_receipts=available_input_receipts,
    )
    return unfilled


def _late_input_column_readiness_rows(
    frame: Frame,
    *,
    input_column: ProducerInputColumn,
    required_scope: str,
    producer_name: str,
    available_input_receipts: Mapping[str, Mapping[str, object]],
) -> tuple[int, int]:
    """Return ``(missing_rows, invalid_values)`` for one declared column."""

    if input_column.entity == "frame":
        if not input_column.column.startswith("@"):
            raise ValueError(
                "Frame-level late inputs require an @ name; "
                f"got {input_column.column!r}."
            )
        metadata_key = input_column.column.removeprefix("@")
        return (
            (0, 0) if isinstance(frame.metadata.get(metadata_key), Mapping) else (1, 0)
        )
    table = frame.table(input_column.entity)
    scope = _late_required_scope_mask(
        frame,
        entity=input_column.entity,
        required_scope=required_scope,
    )
    if input_column.column == "@resolved_weight":
        weights = np.asarray(
            frame.resolve_weights(input_column.entity).values,
            dtype=np.float64,
        )
        if weights.shape != (len(table),):
            return 0, int(scope.sum())
        return 0, int((~np.isfinite(weights) & scope.to_numpy(dtype=bool)).sum())
    if input_column.column.startswith("@"):
        receipt_key = f"{input_column.entity}.{input_column.column}"
        receipt = available_input_receipts.get(receipt_key)
        if _late_available_input_receipt_is_valid(
            receipt,
            producer=producer_name,
            entity=input_column.entity,
            column=input_column.column,
        ):
            return 0, 0
        return max(1, int(scope.sum())), 0
    if input_column.column not in table:
        return int(scope.sum()), 0
    values = table[input_column.column]
    if input_column.value_kind == "column_present":
        return 0, 0
    missing = values.isna()
    invalid = pd.Series(False, index=values.index, dtype=bool)
    if input_column.value_kind == "finite_numeric":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        invalid = (~missing) & ~np.isfinite(numeric)
    return int((missing & scope).sum()), int((invalid & scope).sum())


def _late_declared_absence_receipts(
    contract: ProducerContract,
    unfilled_rows: Mapping[ProducerInput, int],
    *,
    invalid_rows: Mapping[ProducerInput, int],
) -> dict[str, Mapping[str, object]]:
    """Materialize only absences explicitly tolerated by the contract."""

    receipts: dict[str, Mapping[str, object]] = {}
    for requirement, rows in unfilled_rows.items():
        if (
            rows <= 0
            or invalid_rows.get(requirement, 0) > 0
            or not requirement.tolerated_absence_receipts
        ):
            continue
        for receipt_id in requirement.tolerated_absence_receipts:
            receipts[receipt_id] = {
                "receipt_id": receipt_id,
                "status": "declared_absence",
                "entity": requirement.entity,
                "column": requirement.column,
                "required_scope": requirement.required_scope,
                "rows": rows,
                "producer": contract.name,
                "reason": "optional availability-pattern input",
            }
    return receipts


def _assert_primary_puf_stage_complete(frame: Frame) -> None:
    """Validate the already-executed root producer before DAG dispatch."""

    contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[US_LATE_PRIMARY_PUF_STAGE]
    failures: list[str] = []
    for output in contract.outputs:
        if output.entity == "frame":
            metadata_key = output.column.removeprefix("@")
            if not isinstance(frame.metadata.get(metadata_key), Mapping):
                failures.append(
                    f"frame.{output.column}: metadata receipt absent on "
                    f"{output.coverage_scope}"
                )
            continue
        table = frame.table(output.entity)
        scope = _late_required_scope_mask(
            frame,
            entity=output.entity,
            required_scope=output.coverage_scope,
        )
        if output.column == "@resolved_weight":
            weights = np.asarray(
                frame.resolve_weights(output.entity).values,
                dtype=np.float64,
            )
            if weights.shape != (len(table),):
                failures.append(
                    f"{output.entity}.@resolved_weight: shape {weights.shape} "
                    f"does not match {(len(table),)}"
                )
            else:
                count = int(
                    (
                        ~np.isfinite(weights) & scope.to_numpy(dtype=bool, copy=False)
                    ).sum()
                )
                if count:
                    failures.append(
                        f"{output.entity}.@resolved_weight: {count} invalid "
                        f"row(s) on {output.coverage_scope}"
                    )
            continue
        if output.column not in table:
            failures.append(
                f"{output.entity}.{output.column}: column absent on "
                f"{output.coverage_scope}"
            )
            continue
        values = table[output.column]
        missing = values.isna()
        if pd.api.types.is_numeric_dtype(values.dtype):
            missing |= ~np.isfinite(
                pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
            )
        count = int((missing & scope).sum())
        if count:
            failures.append(
                f"{output.entity}.{output.column}: {count} unfilled row(s) on "
                f"{output.coverage_scope}"
            )
    if failures:
        raise ValueError(
            f"Late producer {US_LATE_PRIMARY_PUF_STAGE!r} is not complete:\n  "
            + "\n  ".join(failures)
        )


def _materialize_stacked_acs_earnings_universe(
    frame: Frame,
    *,
    execution_config: Mapping[str, object] | None = None,
) -> AcsPumsEarningsUniverseApplication:
    """Run and bind the declared pre-primary ACS earnings-universe producer."""

    resolved_config = (
        _late_acs_earnings_universe_execution_binding()
        if execution_config is None
        else _json_ready(execution_config)
    )
    columns = resolved_config.get("ordered_mapped_columns")
    if not isinstance(columns, list) or any(
        not isinstance(column, str) or not column for column in columns
    ):
        raise ValueError("Late ACS earnings-universe execution config is malformed.")
    application = apply_acs_pums_earnings_universe_zeros(
        frame,
        columns=tuple(columns),
        boundary="late ACS PUMS earnings-universe producer",
    )
    receipt = _json_ready(application.receipt)
    contract = resolved_config.get("contract_identity")
    if not isinstance(contract, Mapping) or any(
        receipt.get(field) != contract.get(field)
        for field in (
            "version",
            "source_channel",
            "minimum_age",
            "aggregation",
            "produced_frame_semantics",
        )
    ):
        raise ValueError(
            "Late ACS earnings-universe application receipt differs from its "
            "bound runtime contract."
        )
    receipt_rules = [
        {
            "rule_id": rule.get("rule_id"),
            "source_column": rule.get("source_column"),
            "mapped_column": rule.get("mapped_column"),
        }
        for rule in receipt.get("rules", {}).values()
    ]
    if receipt_rules != contract.get("rules"):
        raise ValueError(
            "Late ACS earnings-universe application rules differ from their "
            "bound runtime contract."
        )
    metadata_key = US_LATE_ACS_EARNINGS_UNIVERSE_RECEIPT_INPUT.removeprefix("@")
    if metadata_key in application.frame.metadata:
        raise ValueError(
            "Late ACS PUMS earnings-universe producer found a pre-existing "
            "application receipt."
        )
    bound = Frame(
        {
            entity: application.frame.table(entity)
            for entity in application.frame.entities
        },
        application.frame.schema,
        {
            entity: application.frame.weights_for(entity)
            for entity in application.frame.weighted_entities
        },
        application.frame.strata,
        mass_log=application.frame.mass_log,
        metadata={**application.frame.metadata, metadata_key: receipt},
    )
    return AcsPumsEarningsUniverseApplication(bound, receipt)


def _aggregate_late_transfer_result(
    frame: Frame,
    *,
    group_results: Mapping[
        str,
        tuple[
            Mapping[str, object],
            tuple[object, ...],
            tuple[FitWeightRecord, ...],
            tuple[str, ...],
            str | None,
        ],
    ],
    execution_order: Sequence[str],
) -> StackedPostPufTransferResult:
    """Bind all bounded group outcomes into the canonical 70-target receipt."""

    expected_groups = tuple(group.name for group in CANONICAL_US_LATE_TRANSFER_GROUPS)
    if set(group_results) != set(expected_groups):
        raise ValueError(
            "US late-transfer finalization requires every canonical group once; "
            f"missing={sorted(set(expected_groups) - set(group_results))}, "
            f"extra={sorted(set(group_results) - set(expected_groups))}."
        )
    authority = _production_stacked_authority()
    canonical_family = {
        (entity, target): family
        for entity, families in authority.post_puf_transfer_surface.items()
        for family, targets in families.items()
        for target in targets
    }
    aggregate_targets: dict[str, object] = {}
    imputed_inputs = []
    fit_records = []
    deferred_inputs: list[str] = []
    resolved_channels: set[str | None] = set()
    group_receipts: dict[str, object] = {}
    residual_null_rows = 0
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        (
            group_receipt,
            group_imputed_inputs,
            group_fit_records,
            group_deferred_inputs,
            resolved_donor_channel,
        ) = group_results[group.name]
        if group_receipt.get("producer") != group.name:
            raise ValueError(
                f"US late-transfer group receipt for {group.name!r} is misbound."
            )
        raw_targets = group_receipt.get("targets")
        if not isinstance(raw_targets, Mapping):
            raise ValueError(
                f"US late-transfer group {group.name!r} has no target receipts."
            )
        expected_target_labels = {
            f"{group.entity}/{group.family}/{target}" for target in group.targets
        }
        if set(raw_targets) != expected_target_labels:
            raise ValueError(
                f"US late-transfer group {group.name!r} target receipt drift; "
                f"expected={sorted(expected_target_labels)}, "
                f"got={sorted(raw_targets)}."
            )
        for target in group.targets:
            bounded_label = f"{group.entity}/{group.family}/{target}"
            family = canonical_family[(group.entity, target)]
            aggregate_targets[f"{group.entity}/{family}/{target}"] = dict(
                raw_targets[bounded_label]
            )
            table = frame.table(group.entity)
            if target not in table:
                residual_null_rows += len(table)
            else:
                residual_null_rows += int(table[target].isna().sum())
        group_receipts[group.name] = dict(group_receipt)
        imputed_inputs.extend(group_imputed_inputs)
        fit_records.extend(group_fit_records)
        deferred_inputs.extend(group_deferred_inputs)
        resolved_channels.add(resolved_donor_channel)
    if residual_null_rows:
        raise ValueError(
            "US late-transfer DAG finalization found "
            f"{residual_null_rows} residual null target cell(s); zero are allowed."
        )
    if len(resolved_channels) != 1:
        raise ValueError(
            "US late-transfer groups disagree on resolved donor channel: "
            f"{sorted(map(str, resolved_channels))}."
        )
    aggregate = AcsTransferResult(
        frame=frame,
        imputed_inputs=tuple(imputed_inputs),
        fit_records=tuple(fit_records),
        deferred_inputs=tuple(dict.fromkeys(deferred_inputs)),
        resolved_donor_channel=next(iter(resolved_channels)),
    )
    receipt = {
        "authority": _authority_receipt(authority),
        "producer_schedule": dict(us_late_producer_schedule_receipt()),
        "producer_execution_order": list(execution_order),
        "donor_selection": "owner_projection_of_asec_origin_clone_1",
        "donor_channel": BASE_ASEC_SUPPORT_CHANNEL,
        "donor_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
        "recipient_selection": ("target_specific_complement_of_declared_producer_rows"),
        "resolved_donor_channel": aggregate.resolved_donor_channel,
        "groups": group_receipts,
        "targets": aggregate_targets,
        "fit_records": [
            {"fit_name": record.fit_name, "weight_kind": record.weight_kind}
            for record in aggregate.fit_records
        ],
        "completion": {
            "status": "complete",
            "group_count": len(expected_groups),
            "target_count": len(aggregate_targets),
            "residual_null_rows": 0,
        },
    }
    validate_stacked_post_puf_transfer_receipt(
        receipt,
        boundary="US late-transfer DAG finalization",
    )
    return StackedPostPufTransferResult(frame, receipt, aggregate)


def run_stacked_late_producer_dag(
    frame: Frame,
    *,
    primary_puf_producer: Callable[[Frame], StackedPufPassResult],
    primary_resource_receipts: Mapping[str, Mapping[str, object]],
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_banks: Mapping[str, AcsTransferTargetBank | None] | None = None,
    absence_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> StackedLateProducerResult:
    """Derive and execute the complete late stage from producer contracts."""

    from microcosm.build.us_runtime.multispine_pool import (
        POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
        finalize_multispine_source_inputs,
        run_multispine_post_clone_source_operator,
    )

    if max_targets_per_fit != DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT:
        raise ValueError(
            "Canonical US late-producer groups require "
            f"max_targets_per_fit={DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT}; "
            f"got {max_targets_per_fit}."
        )
    if not callable(primary_puf_producer):
        raise TypeError("US late-producer DAG requires a primary-PUF callback.")
    if not isinstance(primary_resource_receipts, Mapping):
        raise TypeError(
            "US late-producer DAG primary resource receipts must be a mapping."
        )
    expected_primary_resources = _late_contract_available_input_keys(
        CANONICAL_US_LATE_PRODUCER_REGISTRY[US_LATE_PRIMARY_PUF_STAGE]
    )
    if set(primary_resource_receipts) != expected_primary_resources:
        raise ValueError(
            "US late-producer DAG primary resource receipts must exactly cover "
            "the declared virtual inputs; "
            f"missing={sorted(expected_primary_resources - set(primary_resource_receipts))}, "
            f"extra={sorted(set(primary_resource_receipts) - expected_primary_resources)}."
        )
    if US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY in frame.metadata:
        raise ValueError(
            "US late-producer DAG entry already carries a transition authority; "
            "refusing to execute the transition twice."
        )
    expected_groups = {group.name for group in CANONICAL_US_LATE_TRANSFER_GROUPS}
    banks = {} if target_banks is None else dict(target_banks)
    if target_banks is not None and set(banks) != expected_groups:
        raise ValueError(
            "US late-producer target-bank mapping must exactly cover the "
            f"canonical groups; missing={sorted(expected_groups - set(banks))}, "
            f"extra={sorted(set(banks) - expected_groups)}."
        )
    declared_absence = {} if absence_receipts is None else dict(absence_receipts)
    current = frame
    input_frame_sha256 = _late_frame_content_sha256(frame)
    schedule_receipt = _json_ready(us_late_producer_schedule_receipt())
    previous_execution_sha256 = _late_execution_genesis_sha256(
        producer_schedule_sha256=schedule_receipt["payload_sha256"],
        input_frame_sha256=input_frame_sha256,
    )
    execution_order: list[str] = []
    execution_receipts: list[dict[str, object]] = []
    primary_puf_result: StackedPufPassResult | None = None
    source_receipts: dict[str, Mapping[str, object]] = {}
    source_completion_receipt: Mapping[str, object] | None = None
    group_results: dict[
        str,
        tuple[
            Mapping[str, object],
            tuple[object, ...],
            tuple[FitWeightRecord, ...],
            tuple[str, ...],
            str | None,
        ],
    ] = {}
    group_by_name = {group.name: group for group in CANONICAL_US_LATE_TRANSFER_GROUPS}
    for schedule_index, producer_name in enumerate(
        CANONICAL_US_LATE_PRODUCER_SCHEDULE.order
    ):
        contract = CANONICAL_US_LATE_PRODUCER_REGISTRY[producer_name]
        if producer_name == US_LATE_ACS_EARNINGS_UNIVERSE_STAGE:
            node_available_inputs = _late_acs_earnings_universe_resource_receipts()
        elif producer_name == US_LATE_PRIMARY_PUF_STAGE:
            node_available_inputs = dict(primary_resource_receipts)
        elif contract.kind == "post_clone_source":
            node_available_inputs = _late_source_resource_receipts(
                producer_name=producer_name,
            )
        elif producer_name == US_LATE_SOURCE_FINALIZER_STAGE:
            node_available_inputs = _late_source_finalizer_resource_receipts()
            node_available_inputs.update(
                {
                    f"person.@source_receipt:{operator}": (
                        _late_available_input_receipt(
                            producer=US_LATE_SOURCE_FINALIZER_STAGE,
                            entity="person",
                            column=f"@source_receipt:{operator}",
                            rows=len(current.table("person")),
                            binding={
                                "resource_kind": "source_operator_receipt",
                                "schema_version": 1,
                                "source_operator": operator,
                                "source_receipt_sha256": _canonical_sha256(
                                    _json_ready(source_receipts[operator])
                                ),
                            },
                        )
                    )
                    for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
                    if operator in source_receipts
                }
            )
        elif contract.kind == "late_transfer":
            group = group_by_name[producer_name]
            node_available_inputs = _late_transfer_resource_receipts(
                group_name=group.name,
                entity=group.entity,
                family=group.family,
                targets=group.targets,
                seed=seed,
                n_estimators=n_estimators,
                max_targets_per_fit=max_targets_per_fit,
                target_bank=banks.get(producer_name),
            )
        else:
            node_available_inputs = {}
        unfilled_rows, invalid_rows = _late_input_readiness_rows(
            current,
            contract,
            available_input_receipts=node_available_inputs,
        )
        node_absence_receipts = _late_declared_absence_receipts(
            contract,
            unfilled_rows,
            invalid_rows=invalid_rows,
        )
        for receipt_id, receipt in node_absence_receipts.items():
            previous = declared_absence.setdefault(receipt_id, receipt)
            if dict(previous) != dict(receipt):
                raise ValueError(
                    f"Late producer {producer_name!r} absence receipt "
                    f"{receipt_id!r} conflicts with supplied evidence."
                )
        declared_input_evidence = _late_declared_input_evidence(
            current,
            contract,
            available_input_receipts=node_available_inputs,
            unfilled_rows=unfilled_rows,
            invalid_rows=invalid_rows,
        )
        outcome: dict[str, object] = {}

        def execute(
            *,
            bound_contract: ProducerContract = contract,
            bound_producer_name: str = producer_name,
            bound_frame: Frame = current,
            bound_outcome: dict[str, object] = outcome,
            bound_available_inputs: Mapping[
                str, Mapping[str, object]
            ] = node_available_inputs,
        ) -> None:
            if bound_contract.kind == "acs_earnings_universe":
                config_receipt = bound_available_inputs[
                    f"person.{US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT}"
                ]
                result = _materialize_stacked_acs_earnings_universe(
                    bound_frame,
                    execution_config=config_receipt["binding"],
                )
            elif bound_contract.kind == "primary_puf":
                result = primary_puf_producer(bound_frame)
            elif bound_contract.kind == "post_clone_source":
                operator = bound_producer_name.removeprefix("source:")
                result = run_multispine_post_clone_source_operator(
                    bound_frame,
                    operator,
                )
            elif bound_contract.kind == "source_finalizer":
                result = finalize_multispine_source_inputs(
                    bound_frame,
                    operator_receipts=source_receipts,
                )
            elif bound_contract.kind == "late_transfer":
                group = group_by_name[bound_producer_name]
                config_receipt = bound_available_inputs[
                    f"{group.entity}.{US_LATE_TRANSFER_MODEL_CONFIG_INPUT}"
                ]
                config_binding = config_receipt["binding"]
                assert isinstance(config_binding, Mapping)
                execution_contract = config_binding["transfer_execution_contract"]
                assert isinstance(execution_contract, Mapping)
                result = transfer_stacked_post_puf_group(
                    bound_frame,
                    group_name=bound_producer_name,
                    seed=seed,
                    n_estimators=n_estimators,
                    max_targets_per_fit=max_targets_per_fit,
                    target_bank=banks.get(bound_producer_name),
                    execution_contract=execution_contract,
                )
            else:
                raise AssertionError(
                    f"Unhandled US late-producer kind {bound_contract.kind!r}."
                )
            bound_outcome["result"] = result

        run_producer_when_ready(
            contract,
            execute,
            unfilled_rows=unfilled_rows,
            invalid_rows=invalid_rows,
            absence_receipts=declared_absence,
        )
        result = outcome["result"]
        if contract.kind == "acs_earnings_universe":
            _assert_late_callback_consumed_bound_config(
                producer=producer_name,
                entity="person",
                column=US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT,
                available_input_receipts=node_available_inputs,
                actual_binding=_late_acs_earnings_universe_execution_binding(),
            )
        elif contract.kind == "post_clone_source":
            _assert_late_callback_consumed_bound_config(
                producer=producer_name,
                entity="person",
                column=US_LATE_SOURCE_EXECUTION_CONFIG_INPUT,
                available_input_receipts=node_available_inputs,
                actual_binding=_late_source_execution_config_binding(producer_name),
            )
        elif contract.kind == "source_finalizer":
            _assert_late_callback_consumed_bound_config(
                producer=producer_name,
                entity="person",
                column=US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT,
                available_input_receipts=node_available_inputs,
                actual_binding=_late_source_finalizer_execution_binding(),
            )
        elif contract.kind == "late_transfer":
            group = group_by_name[producer_name]
            _assert_late_callback_consumed_bound_config(
                producer=producer_name,
                entity=group.entity,
                column=US_LATE_TRANSFER_MODEL_CONFIG_INPUT,
                available_input_receipts=node_available_inputs,
                actual_binding=_late_transfer_model_config_binding(
                    group_name=group.name,
                    entity=group.entity,
                    family=group.family,
                    targets=group.targets,
                    seed=seed,
                    n_estimators=n_estimators,
                    max_targets_per_fit=max_targets_per_fit,
                ),
            )
            _assert_late_callback_consumed_bound_config(
                producer=producer_name,
                entity=group.entity,
                column=US_LATE_TRANSFER_TARGET_BANK_INPUT,
                available_input_receipts=node_available_inputs,
                actual_binding=_late_transfer_target_bank_binding(
                    group_name=group.name,
                    target_bank=banks.get(producer_name),
                ),
            )
        current = result.frame
        _validate_late_callback_output_metric_families(current, contract)
        producer_receipt = _json_ready(result.receipt)
        if contract.kind == "primary_puf":
            _validate_primary_callback_resource_binding(
                producer_receipt,
                available_input_receipts=node_available_inputs,
                boundary=f"late producer {producer_name!r}",
            )
        output_surface = [
            _late_output_column_evidence(
                current,
                output=output,
                producer_receipt=producer_receipt,
            )
            for output in contract.outputs
        ]
        absent_outputs = [
            f"{item['entity']}.{item['column']}"
            for item in output_surface
            if item.get("status") != "present"
        ]
        if absent_outputs:
            raise ValueError(
                f"Late producer {producer_name!r} completed without declared "
                f"output(s) {absent_outputs}."
            )
        execution_row: dict[str, object] = {
            "execution_index": schedule_index,
            "producer": producer_name,
            "kind": contract.kind,
            "declared_inputs": declared_input_evidence,
            "declared_absence_receipts": {
                receipt_id: dict(receipt)
                for receipt_id, receipt in node_absence_receipts.items()
            },
            "available_input_receipts": {
                receipt_id: dict(receipt)
                for receipt_id, receipt in sorted(node_available_inputs.items())
            },
            "input_surface_sha256": _canonical_sha256(declared_input_evidence),
            "output_surface": output_surface,
            "output_surface_sha256": _canonical_sha256(output_surface),
            "producer_receipt": producer_receipt,
            "producer_receipt_sha256": _canonical_sha256(producer_receipt),
            "previous_execution_sha256": previous_execution_sha256,
            "status": "complete",
        }
        execution_row["sha256"] = _canonical_sha256(execution_row)
        previous_execution_sha256 = str(execution_row["sha256"])
        execution_receipts.append(execution_row)
        if contract.kind == "acs_earnings_universe":
            execution_order.append(producer_name)
            continue
        if contract.kind == "primary_puf":
            if not isinstance(result, StackedPufPassResult):
                raise TypeError(
                    "US primary-PUF producer callback must return StackedPufPassResult."
                )
            _assert_primary_puf_stage_complete(current)
            primary_puf_result = result
            continue

        execution_order.append(producer_name)
        if contract.kind == "post_clone_source":
            operator = producer_name.removeprefix("source:")
            source_receipts[operator] = result.receipt
        elif contract.kind == "source_finalizer":
            source_completion_receipt = result.receipt
        else:
            group = group_by_name[producer_name]
            if tuple(result.receipt.get("ordered_targets", ())) != group.targets:
                raise ValueError(
                    f"Late-transfer producer {producer_name!r} changed its "
                    "declared target order."
                )
            group_results[producer_name] = (
                result.receipt,
                result.transfer_result.imputed_inputs,
                result.transfer_result.fit_records,
                result.transfer_result.deferred_inputs,
                result.transfer_result.resolved_donor_channel,
            )
    if source_completion_receipt is None:
        raise AssertionError("US late-producer DAG did not finalize source inputs.")
    if primary_puf_result is None:
        raise AssertionError("US late-producer DAG did not execute primary PUF.")
    aggregate = _aggregate_late_transfer_result(
        current,
        group_results=group_results,
        execution_order=execution_order,
    )
    late_receipt: dict[str, object] = {
        "version": US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "producer_schedule": schedule_receipt,
        "input_frame_sha256": input_frame_sha256,
        "output_frame_sha256": _late_frame_content_sha256(aggregate.frame),
        "execution_chain_sha256": previous_execution_sha256,
        "execution": execution_receipts,
        "source_completion": _json_ready(source_completion_receipt),
        "post_puf_transfer": _json_ready(aggregate.receipt),
    }
    late_receipt["sha256"] = _canonical_sha256(late_receipt)
    validate_stacked_late_producer_receipt(
        late_receipt,
        boundary="US late-producer DAG finalization",
    )
    authorized_frame, transition_authority_sha256 = (
        _bind_late_producer_transition_authority(aggregate.frame, late_receipt)
    )
    validate_stacked_late_producer_receipt(
        late_receipt,
        boundary="US late-producer DAG live-output finalization",
        frame=authorized_frame,
        expected_transition_authority_sha256=transition_authority_sha256,
    )
    return StackedLateProducerResult(
        frame=authorized_frame,
        receipt=late_receipt,
        primary_puf_result=primary_puf_result,
        source_completion_receipt=source_completion_receipt,
        transfer_result=aggregate.transfer_result,
        transition_authority_sha256=transition_authority_sha256,
    )


# ---------------------------------------------------------------------------
# The single PUF pass over the stacked spine (charter item 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackedPufPassResult:
    """The post-PUF stacked frame plus attachment and fit receipts."""

    frame: Frame
    receipt: Mapping[str, object]


def run_stacked_puf_pass(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    *,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    predictors: Sequence[str] | None = None,
    person_outputs: Sequence[str] | None = None,
    tax_unit_outputs: Sequence[str] | None = None,
    seed: int = 0,
    n_estimators: int = 100,
    fit_records: list[FitWeightRecord] | None = None,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
    primary_qrf_checkpoint_dir: str | Path | None = None,
    primary_qrf_input_binding: Mapping[str, object] | None = None,
) -> StackedPufPassResult:
    """Run the resumable primary QRF and clone-2 tail over the stacked spine.

    Order is the charter's: the spine must already be gap-filled (this entry
    validates the stacked manifest and refuses cloned input), the PUF clone
    arm attaches to a seeded whole-household sample of stacked households
    (both origins; reusing the reviewed clone-routing discipline), and the
    primary QRF then runs under both stacked doctrines — recipient predictors
    must be complete (no zero-filled absence) and finalization preserves
    nulls on every cell the pass does not own.
    """

    return _run_stacked_puf_pass_evaluate(
        frame,
        donor_tax_units,
        clone_attachment_fraction=clone_attachment_fraction,
        clone_attachment_seed=clone_attachment_seed,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
        seed=seed,
        n_estimators=n_estimators,
        fit_records=fit_records,
        tail_bound_diagnostics=tail_bound_diagnostics,
        primary_qrf_checkpoint_dir=primary_qrf_checkpoint_dir,
        primary_qrf_input_binding=primary_qrf_input_binding,
        apply_capital_gains_tail=True,
    )


def _run_stacked_puf_pass_without_tail_for_test(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    **kwargs: object,
) -> StackedPufPassResult:
    """Fixture-only pilot seam; production always applies the clone-2 tail."""

    return _run_stacked_puf_pass_evaluate(
        frame,
        donor_tax_units,
        apply_capital_gains_tail=False,
        **kwargs,
    )


def _run_stacked_puf_pass_evaluate(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    *,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    predictors: Sequence[str] | None = None,
    person_outputs: Sequence[str] | None = None,
    tax_unit_outputs: Sequence[str] | None = None,
    seed: int = 0,
    n_estimators: int = 100,
    fit_records: list[FitWeightRecord] | None = None,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
    primary_qrf_checkpoint_dir: str | Path | None = None,
    primary_qrf_input_binding: Mapping[str, object] | None = None,
    apply_capital_gains_tail: bool,
) -> StackedPufPassResult:
    """Internal evaluator with one explicit fixture-only tail seam."""

    if primary_qrf_input_binding is not None:
        noncanonical = {
            "predictors": predictors,
            "person_outputs": person_outputs,
            "tax_unit_outputs": tax_unit_outputs,
        }
        explicit = sorted(
            name for name, value in noncanonical.items() if value is not None
        )
        if explicit:
            raise ValueError(
                "Stacked production primary QRF requires the import-declared "
                f"canonical predictor/output surface; explicit={explicit}."
            )
        missing_sinks = [
            name
            for name, value in {
                "fit_records": fit_records,
                "tail_bound_diagnostics": tail_bound_diagnostics,
            }.items()
            if value is None
        ]
        if missing_sinks:
            raise ValueError(
                "Stacked production primary QRF requires its declared audit "
                f"sink(s): {missing_sinks}."
            )
    canonical_tail_bounds = (
        puf_tax_detail_tail_bound_quantiles_identity()
        if person_outputs is None and tax_unit_outputs is None
        else None
    )
    tail_spec, tail_agi_bands = resolve_puf_capital_gains_tail_execution_inputs()
    validate_stacked_spine_frame(frame, boundary="stacked PUF pass entry")
    person_clone = frame.table("person")[support_clone_index_column("person")]
    if not person_clone.eq(0).all():
        raise ValueError(
            "The stacked PUF pass owns clone attachment; found nonzero person "
            "support clone indices on its input."
        )
    universe_metadata_key = US_LATE_ACS_EARNINGS_UNIVERSE_RECEIPT_INPUT.removeprefix(
        "@"
    )
    universe_application_receipt = frame.metadata.get(universe_metadata_key)
    if not isinstance(universe_application_receipt, Mapping):
        raise ValueError(
            "Stacked PUF pass requires the declared "
            f"{US_LATE_ACS_EARNINGS_UNIVERSE_STAGE!r} producer receipt before "
            "the primary callback may run."
        )
    resolved_universe = resolve_acs_pums_earnings_universe(
        frame,
        columns=tuple(ACS_PUMS_EARNINGS_SOURCE_COLUMNS),
        boundary="stacked PUF pass ACS earnings-universe receipt",
    )
    if _json_ready(universe_application_receipt) != _json_ready(
        resolved_universe.receipt
    ):
        raise ValueError(
            "Stacked PUF pass ACS earnings-universe receipt does not match "
            "the live produced frame."
        )
    cloned = clone_us_frame_for_puf_support(
        frame,
        channels=(BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL),
        clone_attachment_fraction=clone_attachment_fraction,
        clone_attachment_seed=clone_attachment_seed,
    )
    attachment = validate_puf_clone_attachment(
        cloned,
        boundary="stacked PUF pass clone attachment",
        expected_fraction=clone_attachment_fraction,
        expected_seed=clone_attachment_seed,
    )

    kwargs: dict[str, object] = {}
    if predictors is not None:
        kwargs["predictors"] = tuple(predictors)
    if person_outputs is not None:
        kwargs["person_outputs"] = tuple(person_outputs)
    if tax_unit_outputs is not None:
        kwargs["tax_unit_outputs"] = tuple(tax_unit_outputs)
    primary_resource_receipts_sha256: str | None = None
    if primary_qrf_checkpoint_dir is None:
        if primary_qrf_input_binding is not None:
            raise ValueError(
                "Stacked monolithic primary QRF cannot carry a checkpoint-input "
                "binding without a checkpoint directory."
            )
        predictor_universe_receipts: list[dict[str, object]] = []
        imputed = impute_us_puf_tax_detail_support(
            cloned,
            donor_tax_units,
            seed=seed,
            n_estimators=n_estimators,
            fit_records=fit_records,
            tail_bound_diagnostics=tail_bound_diagnostics,
            tail_bound_quantiles=canonical_tail_bounds,
            predictor_universe_receipts=predictor_universe_receipts,
            require_complete_recipient_predictors=True,
            absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
            **kwargs,
        )
        if len(predictor_universe_receipts) != 1:
            raise AssertionError(
                "Stacked monolithic PUF pass emitted the wrong number of "
                "recipient predictor universe receipts."
            )
        primary_qrf_receipt: dict[str, object] = {
            "mode": "monolithic",
            "resume_status": "not_applicable",
            "recipient_predictor_universe": predictor_universe_receipts[0],
        }
    else:
        _validate_stacked_late_primary_checkpoint_input_binding(
            primary_qrf_input_binding,
            boundary="stacked primary-QRF checkpoint entry",
        )
        assert isinstance(primary_qrf_input_binding, Mapping)
        normalized_input_binding = _json_ready(primary_qrf_input_binding)
        bound_resources = normalized_input_binding["primary_resource_receipts"]
        assert isinstance(bound_resources, Mapping)
        checkpoint_resource = bound_resources["tax_unit.@primary_qrf_checkpoint"]
        assert isinstance(checkpoint_resource, Mapping)
        checkpoint_binding = checkpoint_resource["binding"]
        assert isinstance(checkpoint_binding, Mapping)
        checkpoint_dir = Path(primary_qrf_checkpoint_dir)
        bound_checkpoint_identity = str(
            checkpoint_binding["checkpoint_identity_sha256"]
        )
        observed_checkpoint_identity = checkpoint_dir.resolve().name
        if observed_checkpoint_identity != bound_checkpoint_identity:
            raise ValueError(
                "Stacked primary-QRF checkpoint directory identity differs "
                "from its bound late-producer resource: "
                f"directory={observed_checkpoint_identity!r}, "
                f"bound={bound_checkpoint_identity!r}."
            )
        actual_resources = stacked_late_primary_resource_receipts(
            donor_tax_units,
            primary_qrf_checkpoint_identity_sha256=observed_checkpoint_identity,
            clone_attachment_fraction=clone_attachment_fraction,
            clone_attachment_seed=clone_attachment_seed,
            seed=seed,
            n_estimators=n_estimators,
            fit_records_enabled=fit_records is not None,
            tail_bound_diagnostics_enabled=(tail_bound_diagnostics is not None),
            predictors=predictors,
            person_outputs=person_outputs,
            tax_unit_outputs=tax_unit_outputs,
            capital_gains_tail_spec=tail_spec,
            capital_gains_tail_agi_bands=tail_agi_bands,
        )
        if _json_ready(actual_resources) != _json_ready(bound_resources):
            raise ValueError(
                "Stacked primary-QRF callback invocation disagrees with its "
                "declared late-producer donor/config resources."
            )
        primary_resource_receipts_sha256 = _canonical_sha256(bound_resources)
        manifest_path = checkpoint_dir / PRIMARY_QRF_MANIFEST_FILENAME
        input_binding_path = checkpoint_dir / _LATE_PRIMARY_QRF_INPUT_BINDING_FILENAME
        if manifest_path.exists():
            if not input_binding_path.is_file():
                raise ValueError(
                    "Stacked primary QRF checkpoint has no late-producer input "
                    f"binding: {input_binding_path}."
                )
            try:
                observed_input_binding = json.loads(
                    input_binding_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Stacked primary QRF checkpoint input binding is unreadable: "
                    f"{input_binding_path}."
                ) from exc
            _validate_stacked_late_primary_checkpoint_input_binding(
                observed_input_binding,
                boundary="stacked primary-QRF checkpoint resume",
            )
            if observed_input_binding != normalized_input_binding:
                raise ValueError(
                    "Stacked primary QRF checkpoint input binding differs from "
                    "the live late-producer donor/config resources; refusing "
                    "stale predictions."
                )
            resume_status = "resumed"
        else:
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise ValueError(
                    "Stacked primary QRF checkpoint directory is nonempty but "
                    f"has no manifest: {checkpoint_dir}."
                )
            initialize_primary_puf_qrf_chain(
                cloned,
                donor_tax_units,
                checkpoint_dir,
                seed=seed,
                n_estimators=n_estimators,
                require_complete_recipient_predictors=True,
                absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
                **kwargs,
            )
            input_binding_bytes = json.dumps(
                normalized_input_binding,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            temporary_binding_path = input_binding_path.with_name(
                f".{input_binding_path.name}.tmp"
            )
            temporary_binding_path.write_bytes(input_binding_bytes)
            temporary_binding_path.replace(input_binding_path)
            resume_status = "initialized"
        predictor_universe_receipt = (
            primary_puf_qrf_recipient_predictor_universe_receipt(checkpoint_dir)
        )
        run_primary_puf_qrf_chain(checkpoint_dir, environment={})
        imputed, weight_kind = finalize_primary_puf_qrf_chain(
            cloned,
            checkpoint_dir,
            tail_bound_diagnostics=tail_bound_diagnostics,
            tail_bound_quantiles=canonical_tail_bounds,
        )
        if fit_records is not None:
            fit_records.append(FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, weight_kind))
        primary_qrf_receipt = {
            "mode": "checkpoint_chain",
            "resume_status": resume_status,
            "checkpoint_manifest": str(manifest_path.resolve()),
            "input_binding_sha256": normalized_input_binding["sha256"],
            "recipient_predictor_universe": predictor_universe_receipt,
        }

    validate_stacked_spine_frame(imputed, boundary="stacked primary PUF output")
    validate_puf_clone_attachment(
        imputed,
        boundary="stacked primary PUF output",
        expected_fraction=clone_attachment_fraction,
        expected_seed=clone_attachment_seed,
    )

    if apply_capital_gains_tail:
        output, tail_receipt = transfer_puf_capital_gains_tail(
            imputed,
            donor_tax_units,
            seed=seed,
            spec=tail_spec,
            agi_bands=tail_agi_bands,
        )
        validate_puf_capital_gains_tail_manifest(tail_receipt)
        # The tail producer creates clone role 2 before its final origin
        # receipt can be added to the tail manifest.  Bind the producer's
        # original manifest into a provisional attachment first, so no
        # unreceipted clone role crosses the stacked validation boundary used
        # to derive that origin receipt.  Rebind once more to the final tail
        # digest below.
        provisional = bind_puf_clone_attachment_tail_descendant(
            output,
            attachment_receipt=attachment,
            tail_manifest=tail_receipt,
        )
        tail_receipt = _bind_stacked_tail_origin_receipt(provisional, tail_receipt)
        output = bind_puf_clone_attachment_tail_descendant(
            output,
            attachment_receipt=attachment,
            tail_manifest=tail_receipt,
        )
        attachment = validate_puf_clone_attachment(
            output,
            boundary="stacked PUF tail descendant",
            expected_fraction=clone_attachment_fraction,
            expected_seed=clone_attachment_seed,
        )
        tail_ceiling = tail_receipt["tail_distribution_receipts"]["frame_after_stage"]
        if not tail_ceiling["positive_mass_five_x_target_exceeded"]:
            raise ValueError(
                "PUF capital-gains tail transfer did not clear its declared "
                "five-times positive-mass target: "
                f"{tail_ceiling['positive_mass_five_x_ceiling']} <= "
                f"{tail_ceiling['positive_mass_five_x_target']}."
            )
        tail_status = "applied"
    else:
        output = imputed
        tail_receipt = None
        tail_status = "fixture_only_skipped"
    validate_stacked_spine_frame(output, boundary="stacked PUF pass output")

    person = output.table("person")
    channel = person[support_channel_column("person")].astype(str)
    clone_index = person[support_clone_index_column("person")]
    recipients_by_origin = {
        origin: int((channel.eq(origin) & clone_index.eq(1)).sum())
        for origin in sorted(channel.unique())
    }
    receipt: dict[str, object] = {
        "acs_earnings_universe_application": _json_ready(universe_application_receipt),
        "clone_attachment": _json_ready(attachment),
        "doctrines": {
            "require_complete_recipient_predictors": True,
            "absent_cells": PUF_ABSENT_CELLS_PRESERVE_NULLS,
        },
        "primary_puf_qrf": primary_qrf_receipt,
        "puf_capital_gains_tail_transfer": tail_receipt,
        "tail_status": tail_status,
        "recipient_person_rows_by_origin": recipients_by_origin,
    }
    if primary_resource_receipts_sha256 is not None:
        receipt["primary_resource_receipts_sha256"] = primary_resource_receipts_sha256
    return StackedPufPassResult(frame=output, receipt=receipt)


def _bind_stacked_tail_origin_receipt(
    frame: Frame,
    tail_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Bind clone-2 origin counts at the stacked provenance-owner boundary."""

    validate_stacked_spine_frame(frame, boundary="stacked tail origin binding")
    validate_puf_capital_gains_tail_manifest(tail_manifest)
    household = frame.table("household")
    clone_index = pd.to_numeric(
        household[support_clone_index_column("household")], errors="raise"
    ).astype("int64")
    source_channel_counts = {
        str(channel): int(count)
        for channel, count in sorted(
            household.loc[
                clone_index.eq(2),
                support_channel_column("household"),
            ]
            .astype(str)
            .value_counts()
            .items()
        )
    }
    if not source_channel_counts:
        raise ValueError("Stacked tail origin binding found no clone-2 households.")

    bound = _json_ready(tail_manifest)
    bound.pop("manifest_sha256", None)
    overlap_ownership = _json_ready(us_late_overlap_ownership_receipt())
    existing_overlap_ownership = bound.get("late_overlap_ownership")
    if (
        existing_overlap_ownership is not None
        and existing_overlap_ownership != overlap_ownership
    ):
        raise ValueError(
            "Stacked tail overlap ownership conflicts with the canonical owner matrix."
        )
    bound["late_overlap_ownership"] = overlap_ownership
    clone_receipt = bound.get("clone")
    if not isinstance(clone_receipt, dict):
        raise ValueError("Stacked tail clone provenance receipt is malformed.")
    if clone_receipt.pop("support_channel", None) != (
        PUF_CAPITAL_GAINS_TAIL_SUPPORT_CHANNEL
    ):
        raise ValueError("Stacked tail support-role provenance is malformed.")
    clone_receipt.update(
        {
            "provenance_schema_version": 2,
            "support_role": PUF_CAPITAL_GAINS_TAIL_SUPPORT_CHANNEL,
            "source_channels": source_channel_counts,
        }
    )
    bound["manifest_sha256"] = _canonical_sha256(bound)
    validate_puf_capital_gains_tail_manifest(bound)
    return bound


def prepare_stacked_tail_derivation(frame: Frame) -> tuple[Frame, dict[str, object]]:
    """Clear the clone-2 Schedule-D leaf so derive recomputes it from the tail."""

    validate_stacked_spine_frame(frame, boundary="stacked tail derivation entry")
    person = frame.table("person").copy()
    clone_column = support_clone_index_column("person")
    clone_two = pd.to_numeric(person[clone_column], errors="raise").eq(2)
    if not clone_two.any():
        raise ValueError(
            "Stacked tail derivation requires clone-2 rows from the capital-gains "
            "tail pass."
        )
    column = "schedule_d_capital_gain_distributions"
    if column not in person:
        previously_observed = 0
    else:
        previously_observed = int(person.loc[clone_two, column].notna().sum())
        person.loc[clone_two, column] = np.nan
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    prepared = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return prepared, {
        "column": column,
        "clone_index": 2,
        "cleared_rows": int(clone_two.sum()),
        "previously_observed_rows": previously_observed,
        "column_was_present": column in frame.table("person"),
        "reason": "rederive_from_clone_2_tail_owned_parents",
    }


def assert_stacked_tail_cells_preserved(
    frame: Frame,
    tail_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Prove the complete clone-2 tail state and recipient QRF cells survived."""

    validate_stacked_spine_frame(frame, boundary="stacked tail preservation")
    validate_puf_capital_gains_tail_manifest(tail_manifest)
    overlap_ownership = tail_manifest.get("late_overlap_ownership")
    if not isinstance(overlap_ownership, Mapping):
        raise ValueError(
            "Stacked tail overlap ownership receipt is absent or malformed."
        )
    try:
        overlap_ownership_sha256 = validate_us_late_overlap_ownership_receipt(
            overlap_ownership
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Stacked tail overlap ownership receipt is invalid: {error}"
        ) from error
    attachment = validate_puf_clone_attachment(
        frame,
        boundary="stacked tail preservation attachment",
    )
    transform = attachment.get("post_attachment_transform")
    if not isinstance(transform, Mapping) or transform.get(
        "tail_manifest_sha256"
    ) != tail_manifest.get("manifest_sha256"):
        raise ValueError(
            "Stacked tail preservation attachment is not bound to the supplied "
            "tail manifest."
        )
    records = tail_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Stacked tail preservation requires nonempty records.")

    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    household = frame.table("household")
    person_clone_column = support_clone_index_column("person")
    tax_unit_clone_column = support_clone_index_column("tax_unit")
    household_clone_column = support_clone_index_column("household")
    person_clone = pd.to_numeric(person[person_clone_column], errors="raise").astype(
        "int64"
    )
    tax_unit_clone = pd.to_numeric(
        tax_unit[tax_unit_clone_column], errors="raise"
    ).astype("int64")
    household_clone = pd.to_numeric(
        household[household_clone_column], errors="raise"
    ).astype("int64")

    expected_tail_tax_unit_ids = sorted(
        int(record["tail_tax_unit_id"]) for record in records
    )
    expected_tail_household_ids = sorted(
        int(record["tail_household_id"]) for record in records
    )
    live_tail_tax_unit_ids = sorted(
        tax_unit.loc[tax_unit_clone.eq(2), "tax_unit_id"].astype(int).tolist()
    )
    live_tail_household_ids = sorted(
        household.loc[household_clone.eq(2), "household_id"].astype(int).tolist()
    )
    if live_tail_tax_unit_ids != expected_tail_tax_unit_ids:
        raise ValueError(
            "Stacked tail live clone-2 tax-unit IDs differ from the manifest."
        )
    if live_tail_household_ids != expected_tail_household_ids:
        raise ValueError(
            "Stacked tail live clone-2 household IDs differ from the manifest."
        )
    applied = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN]
    live_applied_ids = sorted(
        tax_unit.loc[applied.eq(True), "tax_unit_id"].astype(int).tolist()  # noqa: E712
    )
    if live_applied_ids != expected_tail_tax_unit_ids:
        raise ValueError(
            "Stacked tail applied-provenance tax-unit IDs differ from the manifest."
        )

    clone_receipt = tail_manifest.get("clone")
    if not isinstance(clone_receipt, Mapping):
        raise ValueError("Stacked tail clone provenance receipt is malformed.")
    if clone_receipt.get("support_role") != PUF_CAPITAL_GAINS_TAIL_SUPPORT_CHANNEL:
        raise ValueError("Stacked tail clone support-role provenance is malformed.")
    live_source_channel_counts = {
        str(channel): int(count)
        for channel, count in sorted(
            household.loc[
                household_clone.eq(2),
                support_channel_column("household"),
            ]
            .astype(str)
            .value_counts()
            .items()
        )
    }
    if clone_receipt.get("source_channels") != live_source_channel_counts:
        raise ValueError(
            "Stacked tail clone source-channel counts differ from the live frame."
        )

    household_weight_by_id = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"].to_numpy(dtype=np.int64),
    )
    tax_unit_weight_by_id = pd.Series(
        frame.resolve_weights("tax_unit").values,
        index=tax_unit["tax_unit_id"].to_numpy(dtype=np.int64),
    )

    observed_state: list[dict[str, object]] = []
    tail_owned_cell_count = 0

    def assert_float_exact(
        actual_value: object,
        expected_value: object,
        *,
        label: str,
    ) -> float:
        expected = np.float64(expected_value)
        actual = np.float64(actual_value)
        if actual.tobytes() != expected.tobytes():
            raise ValueError(
                f"Stacked tail-owned {label} changed: expected {expected!r}, "
                f"got {actual!r}."
            )
        return float(actual)

    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("joint_vector"), Mapping
        ):
            raise ValueError("Stacked tail preservation found a malformed record.")
        joint_vector = record["joint_vector"]
        tail_household_id = int(record["tail_household_id"])
        recipient_household_id = int(record["recipient_household_id"])
        tail_tax_unit_id = int(record["tail_tax_unit_id"])
        recipient_tax_unit_id = int(record["recipient_tax_unit_id"])
        tail_person_id = int(record["tail_person_id"])

        tail_household_rows = household.loc[
            household["household_id"].eq(tail_household_id)
        ]
        recipient_household_rows = household.loc[
            household["household_id"].eq(recipient_household_id)
        ]
        tail_tax_unit_rows = tax_unit.loc[tax_unit["tax_unit_id"].eq(tail_tax_unit_id)]
        recipient_tax_unit_rows = tax_unit.loc[
            tax_unit["tax_unit_id"].eq(recipient_tax_unit_id)
        ]
        for label, rows in (
            ("tail household", tail_household_rows),
            ("recipient household", recipient_household_rows),
            ("tail tax unit", tail_tax_unit_rows),
            ("recipient tax unit", recipient_tax_unit_rows),
        ):
            if len(rows) != 1:
                raise ValueError(
                    f"Stacked tail preservation expected one {label} row; "
                    f"found {len(rows)}."
                )
        tail_household_row = tail_household_rows.iloc[0]
        recipient_household_row = recipient_household_rows.iloc[0]
        tail_tax_unit_row = tail_tax_unit_rows.iloc[0]
        recipient_tax_unit_row = recipient_tax_unit_rows.iloc[0]
        lineage_expectations = (
            (
                "tail household",
                tail_household_row,
                household_clone_column,
                2,
                support_source_id_column("household"),
                record["recipient_household_source_id"],
            ),
            (
                "recipient household",
                recipient_household_row,
                household_clone_column,
                1,
                support_source_id_column("household"),
                record["recipient_household_source_id"],
            ),
            (
                "tail tax unit",
                tail_tax_unit_row,
                tax_unit_clone_column,
                2,
                support_source_id_column("tax_unit"),
                record["recipient_tax_unit_source_id"],
            ),
            (
                "recipient tax unit",
                recipient_tax_unit_row,
                tax_unit_clone_column,
                1,
                support_source_id_column("tax_unit"),
                record["recipient_tax_unit_source_id"],
            ),
        )
        for (
            label,
            row,
            clone_column,
            clone_role,
            source_column,
            source_id,
        ) in lineage_expectations:
            if int(row[clone_column]) != clone_role or int(row[source_column]) != int(
                source_id
            ):
                raise ValueError(f"Stacked tail {label} lineage changed.")

        tail_weight = assert_float_exact(
            household_weight_by_id.loc[tail_household_id],
            record["assigned_weight"],
            label=f"clone-2 household weight for household_id={tail_household_id}",
        )
        recipient_weight = assert_float_exact(
            household_weight_by_id.loc[recipient_household_id],
            record["recipient_household_weight_after"],
            label=(
                "clone-1 residual household weight for "
                f"household_id={recipient_household_id}"
            ),
        )
        assert_float_exact(
            tax_unit_weight_by_id.loc[tail_tax_unit_id],
            record["assigned_weight"],
            label=f"clone-2 tax-unit weight for tax_unit_id={tail_tax_unit_id}",
        )
        assert_float_exact(
            tax_unit_weight_by_id.loc[recipient_tax_unit_id],
            record["recipient_household_weight_after"],
            label=(
                "clone-1 residual tax-unit weight for "
                f"tax_unit_id={recipient_tax_unit_id}"
            ),
        )
        observed_state.extend(
            [
                {
                    "kind": "weight",
                    "role": "tail",
                    "household_id": tail_household_id,
                    "value": tail_weight,
                },
                {
                    "kind": "weight",
                    "role": "recipient",
                    "household_id": recipient_household_id,
                    "value": recipient_weight,
                },
            ]
        )

        provenance_expectations = (
            (PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN, True),
            (
                PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
                int(record["donor_source_id"]),
            ),
            (
                PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
                bool(record["donor_is_synthetic"]),
            ),
            (
                PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
                int(record["donor_filing_status_code"]),
            ),
            (
                PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
                int(record["donor_agi_band_index"]),
            ),
        )
        for column, expected in provenance_expectations:
            actual = tail_tax_unit_row[column]
            if actual != expected or type(actual) is not type(expected):
                # Pandas scalar integer/bool types are valid exact scalar
                # representations even though their Python types differ.
                if isinstance(expected, bool):
                    matches = (
                        isinstance(actual, (bool, np.bool_))
                        and bool(actual) == expected
                    )
                else:
                    matches = (
                        isinstance(actual, (int, np.integer))
                        and int(actual) == expected
                    )
                if not matches:
                    raise ValueError(
                        f"Stacked tail provenance {column} for "
                        f"tax_unit_id={tail_tax_unit_id} changed."
                    )
            observed_state.append(
                {
                    "kind": "provenance",
                    "column": column,
                    "tax_unit_id": tail_tax_unit_id,
                    "value": bool(actual)
                    if isinstance(expected, bool)
                    else int(actual),
                }
            )
        transfer_weight = assert_float_exact(
            tail_tax_unit_row[PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN],
            record["assigned_weight"],
            label=(f"transfer-weight provenance for tax_unit_id={tail_tax_unit_id}"),
        )
        observed_state.append(
            {
                "kind": "provenance",
                "column": PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
                "tax_unit_id": tail_tax_unit_id,
                "value": transfer_weight,
            }
        )

        tail_people = person.loc[person["person_tax_unit_id"].eq(tail_tax_unit_id)]
        if tail_people.empty or not person_clone.loc[tail_people.index].eq(2).all():
            raise ValueError(
                f"Stacked tail tax_unit_id={tail_tax_unit_id} has malformed people."
            )
        if int(tail_people["person_id"].eq(tail_person_id).sum()) != 1:
            raise ValueError(
                f"Stacked tail carrier person_id={tail_person_id} is not unique."
            )
        for _, row in tail_people.iterrows():
            person_id = int(row["person_id"])
            carrier = person_id == tail_person_id
            for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
                if column not in row:
                    raise ValueError(
                        f"Stacked tail-owned column person.{column} is absent."
                    )
                actual = assert_float_exact(
                    row[column],
                    joint_vector[column] if carrier else 0.0,
                    label=f"cell person.{column} for person_id={person_id}",
                )
                observed_state.append(
                    {
                        "kind": "cell",
                        "entity": "person",
                        "column": column,
                        "id": person_id,
                        "value": actual,
                    }
                )
                tail_owned_cell_count += 1
        for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
            if column not in tail_tax_unit_row:
                raise ValueError(
                    f"Stacked tail-owned column tax_unit.{column} is absent."
                )
            actual = assert_float_exact(
                tail_tax_unit_row[column],
                joint_vector[column],
                label=f"cell tax_unit.{column} for tax_unit_id={tail_tax_unit_id}",
            )
            observed_state.append(
                {
                    "kind": "cell",
                    "entity": "tax_unit",
                    "column": column,
                    "id": tail_tax_unit_id,
                    "value": actual,
                }
            )
            tail_owned_cell_count += 1

    preserved_nonowned = 0
    for entity, owned_columns, qrf_outputs in (
        (
            "person",
            frozenset(PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS),
            PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        ),
        (
            "tax_unit",
            frozenset(PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS),
            PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
        ),
    ):
        table = frame.table(entity)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(entity)], errors="raise"
        ).astype("int64")
        # Raw spine IDs may legally collide across ASEC and ACS.  Clone
        # parentage is defined by the assembly-unique pre-clone support ID,
        # which the clone operators preserve across clone roles.
        source_id = support_source_id_column(entity)
        primary = table.loc[clone_index.eq(1)].set_index(source_id, drop=False)
        tail = table.loc[clone_index.eq(2)].set_index(source_id, drop=False)
        missing_sources = sorted(set(tail.index) - set(primary.index))
        if missing_sources:
            raise ValueError(
                f"Stacked tail {entity} clone-2 source rows have no clone-1 "
                f"parent: {missing_sources}."
            )
        for column in sorted(set(qrf_outputs) - owned_columns):
            if column not in table:
                continue
            expected = primary.loc[tail.index, column].reset_index(drop=True)
            actual = tail[column].reset_index(drop=True)
            expected.name = column
            actual.name = column
            if _canonical_donor_series_payload(
                actual,
                boundary=f"stacked tail actual {entity}.{column}",
            ) != _canonical_donor_series_payload(
                expected,
                boundary=f"stacked tail expected {entity}.{column}",
            ):
                raise ValueError(
                    f"Stacked tail recipient-owned QRF column {entity}.{column} "
                    "changed on clone 2."
                )
            preserved_nonowned += int(len(actual))

    return {
        "passed": True,
        "record_count": len(records),
        "tail_owned_cell_count": tail_owned_cell_count,
        "tail_owned_state_count": len(observed_state),
        "recipient_owned_qrf_cell_count": preserved_nonowned,
        "tail_owned_cells_sha256": _canonical_sha256(observed_state),
        "overlap_ownership_sha256": overlap_ownership_sha256,
    }


# ---------------------------------------------------------------------------
# Pre-simulation completeness gate (charter item 4)
# ---------------------------------------------------------------------------

_COMPLETENESS_GATE_NAME = "us_stacked_completeness"
_TAIL_SUPPORT_GATE_DETAIL_KEY = "puf_capital_gains_tail_support"
_ANY_CHANNEL = "*"


def _frame_has_clone_two_rows(frame: Frame) -> bool:
    """Return whether any live entity carries the tail-owned clone role."""

    for entity in frame.entities:
        table = frame.table(entity)
        clone_column = support_clone_index_column(entity)
        if clone_column not in table:
            continue
        clone_index = pd.to_numeric(table[clone_column], errors="raise")
        if clone_index.eq(2).any():
            return True
    return False


def _terminal_tail_support_gate_receipt(
    frame: Frame,
    tail_manifest: Mapping[str, object] | None,
    *,
    boundary: str,
) -> dict[str, object] | None:
    """Authenticate the tail support receipt against the live clone identity."""

    has_clone_two = _frame_has_clone_two_rows(frame)
    if tail_manifest is None:
        if has_clone_two:
            raise ValueError(
                f"{boundary}: live clone-2 rows require the bound PUF "
                "capital-gains-tail manifest."
            )
        return None
    if not has_clone_two:
        raise ValueError(
            f"{boundary}: a supplied PUF capital-gains-tail manifest requires "
            "live clone-2 rows."
        )

    validate_puf_capital_gains_tail_manifest(tail_manifest)
    terminal_receipt = puf_capital_gains_tail_terminal_support_receipt(tail_manifest)
    validate_puf_capital_gains_tail_terminal_support_receipt(terminal_receipt)

    if has_clone_two:
        attachment = validate_puf_clone_attachment(
            frame,
            boundary=f"{boundary} tail attachment",
        )
        transform = attachment.get("post_attachment_transform")
        if not isinstance(transform, Mapping) or transform.get(
            "tail_manifest_sha256"
        ) != tail_manifest.get("manifest_sha256"):
            raise ValueError(
                f"{boundary}: live clone-2 attachment is not bound to the "
                "supplied PUF capital-gains-tail manifest."
            )

    return _json_ready(terminal_receipt)


@dataclass(frozen=True)
class AbsenceProof:
    """An explicit source-by-role authority proof for permitted null cells.

    A declared target's cells may be null only where a proof names the exact
    origin channel and clone role, with the reason recorded.  ``"*"`` may
    cover every origin only when no gap-fill direction declares a donor for
    that target.  This is the audit's item-5 contract: a target is banked or
    imputed, or its absence carries an explicit source-by-role proof — silence
    is never authority.
    """

    entity: str
    column: str
    channel: str
    clone_index: int
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("entity", self.entity),
            ("column", self.column),
            ("channel", self.channel),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"AbsenceProof.{label} must be a non-empty string.")
        if (
            isinstance(self.clone_index, bool)
            or not isinstance(self.clone_index, int)
            or self.clone_index < 0
        ):
            raise ValueError("AbsenceProof.clone_index must be a non-negative integer.")


def _declared_metric_invalidity(
    series: pd.Series,
    *,
    metric: str,
    scope: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return invalid rows under one declared metric, never inferred dtype."""

    if metric not in ORIGIN_BATTERY_METRIC_KINDS:
        raise ValueError(f"Unknown declared metric {metric!r}.")
    if scope.dtype != np.bool_ or scope.ndim != 1 or len(scope) != len(series):
        raise ValueError("Declared-metric validity scope must be an aligned mask.")
    invalid = np.zeros(len(series), dtype=bool)
    positions = np.flatnonzero(scope & ~series.isna().to_numpy(dtype=bool))
    counts = {
        "non_numeric_rows": 0,
        "non_finite_rows": 0,
        "outside_boolean_domain_rows": 0,
        "invalid_categorical_rows": 0,
    }
    if not len(positions):
        return invalid, counts

    scoped = series.iloc[positions]
    if metric == "categorical_tvd":
        local_invalid = np.zeros(len(scoped), dtype=bool)
        for index, value in enumerate(scoped.to_numpy(dtype=object)):
            valid = bool(pd.api.types.is_scalar(value))
            if valid:
                try:
                    hash(value)
                except TypeError:
                    valid = False
            if valid and isinstance(value, (float, np.floating)):
                valid = bool(np.isfinite(value))
            local_invalid[index] = not valid
        invalid[positions] = local_invalid
        counts["invalid_categorical_rows"] = int(local_invalid.sum())
        return invalid, counts

    numeric = pd.to_numeric(scoped, errors="coerce").to_numpy(dtype=np.float64)
    non_numeric = np.isnan(numeric)
    non_finite = np.isinf(numeric)
    local_invalid = non_numeric | non_finite
    counts["non_numeric_rows"] = int(non_numeric.sum())
    counts["non_finite_rows"] = int(non_finite.sum())
    if metric == "boolean_incidence":
        outside_boolean = np.isfinite(numeric) & ~np.isin(numeric, (0.0, 1.0))
        local_invalid |= outside_boolean
        counts["outside_boolean_domain_rows"] = int(outside_boolean.sum())
    invalid[positions] = local_invalid
    return invalid, counts


def _nonzero_invalidity_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """Keep invalidity receipts compact and canonical-JSON safe."""

    return {key: int(value) for key, value in counts.items() if int(value)}


def stacked_completeness_gate(
    frame: Frame,
    *,
    absence_proofs: Sequence[AbsenceProof] = (),
    tail_manifest: Mapping[str, object] | None = None,
) -> GateResult:
    """Evaluate the canonical declared surface with no caller authority."""

    return _stacked_completeness_gate_evaluate(
        frame,
        authority=_production_stacked_authority(),
        production=True,
        absence_proofs=absence_proofs,
        tail_manifest=tail_manifest,
    )


def _stacked_completeness_gate_with_test_authority(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    absence_proofs: Sequence[AbsenceProof] = (),
    tail_manifest: Mapping[str, object] | None = None,
) -> GateResult:
    """Explicit test-only completeness seam for a digested authority bundle."""

    _validate_test_authority(authority, boundary="stacked completeness test seam")
    return _stacked_completeness_gate_evaluate(
        frame,
        authority=authority,
        production=False,
        absence_proofs=absence_proofs,
        tail_manifest=tail_manifest,
    )


def _stacked_completeness_gate_evaluate(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    production: bool,
    absence_proofs: Sequence[AbsenceProof],
    tail_manifest: Mapping[str, object] | None = None,
    _canonical_gap_fill_plan: tuple[
        GapFillDirection, ...
    ] = CANONICAL_STACKED_GAP_FILL_PLAN,
    _canonical_post_puf_surface: TargetFamilies = (
        CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
    ),
) -> GateResult:
    """Prove every declared target is filled or carries absence authority.

    For every declared ``entity/family/target``: a missing column is a named
    terminal failure (a whole registered family can never silently vanish
    again — this is the check that would have caught run 7's 58-target skip);
    a null cell is permitted only where an :class:`AbsenceProof` names its
    origin channel and clone role, and every unproven null fails by name with
    its per-origin, per-role counts. The canonical bundle's live gap-fill plan
    is required authority: its donor direction disables wildcard proofs for
    that target even at the explicitly non-canonical fixture seam.
    """

    tail_support_receipt = _terminal_tail_support_gate_receipt(
        frame,
        tail_manifest,
        boundary="stacked completeness gate",
    )
    authority_receipt = _authority_receipt(authority)
    declared_surface = authority.declared_surface
    declared_count = len(_surface_target_keys(declared_surface))
    failures = _authority_validation_failures(authority, production=production)
    if declared_count == 0:
        failures.append("declared stacked surface contains zero targets.")
    if failures:
        return _sealed_stacked_gate_result(
            name=_COMPLETENESS_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details={
                "authority": authority_receipt,
                "declared_targets": declared_count,
                "targets": {},
                **(
                    {_TAIL_SUPPORT_GATE_DETAIL_KEY: tail_support_receipt}
                    if tail_support_receipt is not None
                    else {}
                ),
            },
        )

    declared_directions: dict[tuple[str, str], GapFillDirection] = {}
    declared_absence_rules: dict[
        tuple[str, str], tuple[GapFillDirection, GapFillAbsenceRule]
    ] = {}
    for direction in authority.gap_fill_plan:
        for entity, targets in _direction_entity_targets(direction).items():
            for target in targets:
                key = (entity, target)
                previous = declared_directions.get(key)
                if previous is not None and previous != direction:
                    raise ValueError(
                        "stacked authority plan assigns conflicting directions "
                        f"to {entity}/{target}: {previous.name!r} and "
                        f"{direction.name!r}."
                    )
                declared_directions[key] = direction
        for rule in direction.recipient_absence_rules:
            declared_absence_rules[(rule.entity, rule.column)] = (direction, rule)
    canonical_direction_keys: set[tuple[str, str]] = set()
    for direction in _canonical_gap_fill_plan:
        for entity, targets in _direction_entity_targets(direction).items():
            for target in targets:
                key = (entity, target)
                canonical_direction_keys.add(key)
                declared_directions[key] = direction
        for rule in direction.recipient_absence_rules:
            declared_absence_rules[(rule.entity, rule.column)] = (direction, rule)
    declared_post_puf_keys = {
        (entity, target)
        for entity, families in authority.post_puf_transfer_surface.items()
        for targets in families.values()
        for target in targets
    }
    canonical_post_puf_keys = {
        (entity, target)
        for entity, families in _canonical_post_puf_surface.items()
        for targets in families.values()
        for target in targets
    }
    post_puf_keys = declared_post_puf_keys | canonical_post_puf_keys

    proof_index: dict[tuple[str, str], dict[tuple[str, int], str]] = {}
    for proof in absence_proofs:
        if not isinstance(proof, AbsenceProof):
            raise ValueError(
                "absence_proofs must contain AbsenceProof values, got "
                f"{type(proof).__name__}."
            )
        proof_index.setdefault((proof.entity, proof.column), {})[
            (proof.channel, proof.clone_index)
        ] = proof.reason

    target_receipts: dict[str, dict[str, object]] = {}
    metric_by_target = {
        (entity, family, column): metric
        for (entity, family, column, _clone_index), metric in (
            authority.metric_registry.items()
        )
    }
    authority_sha256 = authority_receipt["sha256"]
    plan_sha256 = authority_receipt["components"]["gap_fill_plan"]["sha256"]
    post_puf_surface_sha256 = authority_receipt["components"][
        "post_puf_transfer_surface"
    ]["sha256"]
    surface_sha256 = authority_receipt["components"]["declared_surface"]["sha256"]

    def authority_binding(authority_form: str) -> dict[str, object]:
        return {
            "authority_form": authority_form,
            "authority_sha256": authority_sha256,
            "plan_sha256": plan_sha256,
            "post_puf_surface_sha256": post_puf_surface_sha256,
            "surface_sha256": surface_sha256,
        }

    for entity, families in declared_surface.items():
        if entity not in frame.entities:
            failures.append(
                f"{entity}: declared entity is absent from the stacked frame."
            )
            for family, targets in families.items():
                for target in targets:
                    target_receipts[f"{entity}/{family}/{target}"] = {
                        "status": "missing_entity",
                        "null_rows": None,
                        **authority_binding("missing_declared_entity"),
                    }
            continue
        table = frame.table(entity)
        channel = table[support_channel_column(entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(entity)],
            errors="raise",
        ).astype("int64")
        positive_weight = (
            np.asarray(frame.resolve_weights(entity).values, dtype=np.float64) > 0.0
        )
        for family, targets in families.items():
            for target in targets:
                label = f"{entity}/{family}/{target}"
                if target not in table.columns:
                    failures.append(
                        f"{label}: declared target column is missing from the "
                        "pre-simulation pool; the registered family must "
                        "never silently vanish from the active bank."
                    )
                    target_receipts[label] = {
                        "status": "missing",
                        "null_rows": None,
                        **authority_binding("missing_declared_target"),
                    }
                    continue
                null_mask = table[target].isna()
                structural_absence_receipt: dict[str, object] | None = None
                structural_absence_mismatch = False
                structural_rule = declared_absence_rules.get((entity, target))
                if structural_rule is not None:
                    rule_direction, rule = structural_rule
                    structural_mask, structural_absence_receipt = (
                        _gap_fill_absence_rule_mask(
                            frame,
                            direction=rule_direction,
                            rule=rule,
                        )
                    )
                    unexpected = int((null_mask & ~structural_mask).sum())
                    synthesized = int((structural_mask & ~null_mask).sum())
                    structural_absence_receipt = {
                        **structural_absence_receipt,
                        "unexpected_null_rows": unexpected,
                        "structural_rows_filled": synthesized,
                    }
                    structural_absence_mismatch = bool(unexpected or synthesized)
                    if structural_absence_mismatch:
                        failures.append(
                            f"{label}: exact structural-absence equation failed; "
                            f"unexpected_null_rows={unexpected}, "
                            f"structural_rows_filled={synthesized}."
                        )
                metric = metric_by_target[(entity, family, target)]
                invalid_mask, invalidity = _declared_metric_invalidity(
                    table[target],
                    metric=metric,
                    scope=positive_weight,
                )
                invalid_rows = int(invalid_mask.sum())
                invalid_by_origin_role: dict[str, int] = {}
                if invalid_rows:
                    invalid_by_origin_role = {
                        f"{cell_channel}/clone_{int(cell_clone)}": int(count)
                        for (cell_channel, cell_clone), count in (
                            pd.DataFrame(
                                {
                                    "channel": channel.loc[invalid_mask],
                                    "clone_index": clone_index.loc[invalid_mask],
                                }
                            )
                            .groupby(["channel", "clone_index"], sort=True)
                            .size()
                            .items()
                        )
                    }
                    failures.append(
                        f"{label}: declared {metric} metric has {invalid_rows} "
                        "invalid positive-weight value(s), including "
                        f"{_nonzero_invalidity_counts(invalidity)} (by "
                        f"origin/role: {invalid_by_origin_role})."
                    )
                if not null_mask.any():
                    target_receipts[label] = {
                        "status": (
                            "structural_absence_mismatch"
                            if structural_absence_mismatch
                            else "invalid_values"
                            if invalid_rows
                            else "complete"
                        ),
                        "null_rows": 0,
                        "metric": metric,
                        "invalid_rows": invalid_rows,
                        "invalidity": _nonzero_invalidity_counts(invalidity),
                        "invalid_by_origin_role": invalid_by_origin_role,
                        **authority_binding(
                            "invalid_declared_metric_values"
                            if invalid_rows
                            else "observed_complete"
                        ),
                        **(
                            {
                                "recipient_absence_authority": (
                                    structural_absence_receipt
                                )
                            }
                            if structural_absence_receipt is not None
                            else {}
                        ),
                    }
                    continue
                proofs = dict(proof_index.get((entity, target), {}))
                if structural_rule is not None:
                    proofs = {}
                    if not structural_absence_mismatch:
                        _rule_direction, rule = structural_rule
                        structural_mask, _receipt = _gap_fill_absence_rule_mask(
                            frame,
                            direction=_rule_direction,
                            rule=rule,
                        )
                        structural_channels = channel.loc[structural_mask]
                        structural_clones = clone_index.loc[structural_mask]
                        for cell_channel, cell_clone in set(
                            zip(
                                structural_channels.astype(str),
                                structural_clones.astype(int),
                                strict=True,
                            )
                        ):
                            proofs[(str(cell_channel), int(cell_clone))] = rule.reason
                declared_direction = declared_directions.get((entity, target))
                post_puf_target = (entity, target) in post_puf_keys
                null_channels = channel.loc[null_mask]
                null_clones = clone_index.loc[null_mask]
                unproven: dict[str, int] = {}
                proven: dict[str, dict[str, object]] = {}
                grouped = (
                    pd.DataFrame({"channel": null_channels, "clone_index": null_clones})
                    .groupby(["channel", "clone_index"], sort=True)
                    .size()
                )
                target_authority_forms: set[str] = set()
                for (cell_channel, cell_clone), count in grouped.items():
                    cell_channel = str(cell_channel)
                    cell_clone = int(cell_clone)
                    reason = proofs.get((cell_channel, cell_clone))
                    authority_form = "origin_exact"
                    if post_puf_target:
                        if reason is not None or (_ANY_CHANNEL, cell_clone) in proofs:
                            failures.append(
                                f"{label}: absence authority is forbidden because "
                                "the declared post-PUF transfer requires zero "
                                f"residual nulls; found {int(count)} null cell(s) "
                                f"on {cell_channel}/clone_{cell_clone}."
                            )
                        reason = None
                    if declared_direction is not None and reason is not None:
                        if cell_channel != declared_direction.recipient_channel:
                            failures.append(
                                f"{label}: origin-exact authority proof is valid "
                                "only for declared recipient "
                                f"{declared_direction.recipient_channel!r}; "
                                f"{cell_channel!r} is declared donor "
                                f"{declared_direction.donor_channel!r}."
                            )
                            reason = None
                        else:
                            authority_form = "origin_exact_recipient"
                    if (
                        reason is None
                        and declared_direction is None
                        and not post_puf_target
                    ):
                        reason = proofs.get((_ANY_CHANNEL, cell_clone))
                        authority_form = "wildcard_no_declared_donor_plan"
                    key = f"{cell_channel}/clone_{cell_clone}"
                    if reason is None:
                        unproven[key] = int(count)
                        if declared_direction is not None:
                            canonical = (entity, target) in canonical_direction_keys
                            failures.append(
                                f"{label}: {int(count)} null cell(s) on {key} "
                                "require an origin-exact authority proof because "
                                f"{'canonical ' if canonical else ''}gap-fill plan "
                                f"{declared_direction.name!r} names donor "
                                f"{declared_direction.donor_channel!r} and recipient "
                                f"{declared_direction.recipient_channel!r}; wildcard "
                                "authority is forbidden."
                            )
                        elif post_puf_target:
                            failures.append(
                                f"{label}: {int(count)} null cell(s) on {key} "
                                "violate the zero-residual post-PUF transfer "
                                "contract; absence proofs are forbidden."
                            )
                    else:
                        proof_receipt: dict[str, object] = {
                            "null_rows": int(count),
                            "reason": reason,
                            **authority_binding(authority_form),
                        }
                        if declared_direction is not None:
                            proof_receipt.update(
                                {
                                    "declared_direction": declared_direction.name,
                                    "declared_donor_channel": (
                                        declared_direction.donor_channel
                                    ),
                                    "declared_recipient_channel": (
                                        declared_direction.recipient_channel
                                    ),
                                }
                            )
                        if structural_rule is not None:
                            _rule_direction, rule = structural_rule
                            proof_receipt.update(
                                {
                                    "structural_absence_rule_id": rule.rule_id,
                                    "structural_absence_selection": rule.selection,
                                }
                            )
                        proven[key] = proof_receipt
                        target_authority_forms.add(authority_form)
                if unproven:
                    failures.append(
                        f"{label}: {sum(unproven.values())} null cell(s) have "
                        "no source-by-role authority proof "
                        f"(by origin/role: {unproven})."
                    )
                if unproven:
                    target_authority_form = "unproven"
                elif len(target_authority_forms) == 1:
                    target_authority_form = next(iter(target_authority_forms))
                else:
                    target_authority_form = "mixed_proven_absence"
                target_receipts[label] = {
                    "status": (
                        "unproven"
                        if unproven
                        else "invalid_values"
                        if invalid_rows
                        else "proven_absent"
                    ),
                    "null_rows": int(null_mask.sum()),
                    "metric": metric,
                    "invalid_rows": invalid_rows,
                    "invalidity": _nonzero_invalidity_counts(invalidity),
                    "invalid_by_origin_role": invalid_by_origin_role,
                    "proven": proven,
                    "unproven": unproven,
                    **authority_binding(
                        "invalid_declared_metric_values"
                        if invalid_rows and not unproven
                        else target_authority_form
                    ),
                    **(
                        {"recipient_absence_authority": (structural_absence_receipt)}
                        if structural_absence_receipt is not None
                        else {}
                    ),
                }
    return _sealed_stacked_gate_result(
        name=_COMPLETENESS_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={
            "authority": authority_receipt,
            "declared_targets": declared_count,
            "targets": target_receipts,
            **(
                {_TAIL_SUPPORT_GATE_DETAIL_KEY: tail_support_receipt}
                if tail_support_receipt is not None
                else {}
            ),
        },
    )


# ---------------------------------------------------------------------------
# By-origin battery (charter item 5)
# ---------------------------------------------------------------------------

_BATTERY_GATE_NAME = "us_by_origin_battery"
_BATTERY_INCIDENCE_RATIO_BOUNDS = (0.8, 1.25)
_BATTERY_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_BATTERY_QUANTILE_ENVELOPE_TOLERANCE = 0.25
_BATTERY_CATEGORICAL_TVD_TOLERANCE = 0.25


@dataclass(frozen=True)
class OriginBatterySpec:
    """Test-seam grouping for per-column battery metrics.

    Production never accepts these specs from a caller: it consumes the
    immutable 131-column canonical registry. The explicit test-authority seam
    groups its digested registry into specs so the comparison engine can reuse
    the same loop. ``clone_index`` scopes a fixture comparison to one clone
    role: 0 compares native rows and 1 compares a PUF arm.
    """

    entity: str
    family: str
    column_metrics: Mapping[str, str]
    clone_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.entity, str) or not self.entity.strip():
            raise ValueError("OriginBatterySpec.entity must be a non-empty string.")
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("OriginBatterySpec.family must be a non-empty string.")
        if not isinstance(self.column_metrics, Mapping) or not self.column_metrics:
            raise ValueError(
                f"OriginBatterySpec {self.entity}/{self.family} declares no "
                "column metrics."
            )
        unknown = sorted(
            {
                metric
                for metric in self.column_metrics.values()
                if metric not in ORIGIN_BATTERY_METRIC_KINDS
            }
        )
        if unknown:
            raise ValueError(
                f"OriginBatterySpec {self.entity}/{self.family} declares "
                f"unknown metric kind(s) {unknown}; expected one of "
                f"{list(ORIGIN_BATTERY_METRIC_KINDS)}."
            )
        object.__setattr__(
            self,
            "column_metrics",
            MappingProxyType(dict(self.column_metrics)),
        )
        if (
            isinstance(self.clone_index, bool)
            or not isinstance(self.clone_index, int)
            or self.clone_index < 0
        ):
            raise ValueError(
                "OriginBatterySpec.clone_index must be a non-negative integer."
            )


def by_origin_battery(
    frame: Frame,
    *,
    tail_manifest: Mapping[str, object] | None = None,
) -> GateResult:
    """Run the canonical 131-target plus joint by-origin battery."""

    return _by_origin_battery_evaluate(
        frame,
        authority=_production_stacked_authority(),
        production=True,
        tail_manifest=tail_manifest,
    )


def _by_origin_battery_with_test_authority(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    tail_manifest: Mapping[str, object] | None = None,
) -> GateResult:
    """Explicit test-only battery seam for a digested authority bundle."""

    _validate_test_authority(authority, boundary="by-origin battery test seam")
    return _by_origin_battery_evaluate(
        frame,
        authority=authority,
        production=False,
        tail_manifest=tail_manifest,
    )


def _battery_specs_from_metric_registry(
    registry: Mapping[tuple[str, str, str, int], str],
) -> tuple[OriginBatterySpec, ...]:
    grouped: dict[tuple[str, str, int], dict[str, str]] = {}
    for (entity, family, column, clone_index), metric in registry.items():
        grouped.setdefault((entity, family, clone_index), {})[column] = metric
    return tuple(
        OriginBatterySpec(
            entity=entity,
            family=family,
            clone_index=clone_index,
            column_metrics=column_metrics,
        )
        for (entity, family, clone_index), column_metrics in sorted(grouped.items())
    )


def _by_origin_battery_evaluate(
    frame: Frame,
    *,
    authority: _StackedAuthority,
    production: bool,
    tail_manifest: Mapping[str, object] | None = None,
    _canonical_gap_fill_plan: tuple[
        GapFillDirection, ...
    ] = CANONICAL_STACKED_GAP_FILL_PLAN,
) -> GateResult:
    """Compare declared statistics between origins within the one spine.

    Replaces the retired spine-vs-spine agreement: the same comparisons over
    the complete terminal production surface, with per-column DECLARED
    metrics.  The
    tolerances are the chartered ones — incidence ratios within
    ``[0.8, 1.25]``, conditional-quantile envelopes within ``0.25``,
    categorical total-variation within ``0.25`` — deliberately NOT widened.

    Support-awareness is a validity domain, not a tolerance: a comparison
    whose scope rows on either origin fall below the spec's minimum
    effective support is receipted ``insufficient_support`` instead of
    producing a fake confident verdict, and a quantile envelope is evaluated
    only when both origins carry at least that many nonzero rows on the
    compared leg.  A tested rare comparison still fails on any one-sided
    hole — the run-7 ``160,667x`` class fails under every profile because
    both origins carry ample support.
    """

    tail_support_receipt = _terminal_tail_support_gate_receipt(
        frame,
        tail_manifest,
        boundary="by-origin battery",
    )
    authority_receipt = _authority_receipt(authority)
    specs = _battery_specs_from_metric_registry(authority.metric_registry)
    registered_targets = set(authority.metric_registry)
    declared_target_set = set(_surface_target_keys(authority.declared_surface))
    missing_targets = tuple(sorted(declared_target_set - registered_targets))
    extra_targets = tuple(sorted(registered_targets - declared_target_set))
    registration_failures = _authority_validation_failures(
        authority,
        production=production,
    )
    registration_failures.extend(
        f"missing declared battery target {_battery_target_label(target)}."
        for target in missing_targets
    )
    registration_failures.extend(
        f"metric registry target {_battery_target_label(target)} is outside the "
        "declared surface."
        for target in extra_targets
    )
    if not specs:
        registration_failures.append(
            "The by-origin battery requires at least one declared metric."
        )
    support_profile_receipt = authority_receipt["components"]["support_profile"]
    coverage_details = {
        "authority": authority_receipt,
        "declared_target_count": len(declared_target_set),
        "registered_target_count": len(registered_targets),
        "registered_joint_target_count": len(authority.joint_metric_registry),
        "missing_declared_targets": [
            _battery_target_label(target) for target in missing_targets
        ],
        "extra_registered_targets": [
            _battery_target_label(target) for target in extra_targets
        ],
        "declared_plan": {
            "plan_id": "stacked_gap_fill_plan",
            "version": authority.version,
            "sha256": authority_receipt["components"]["gap_fill_plan"]["sha256"],
        },
        "support_profile": support_profile_receipt,
    }
    if registration_failures:
        return _sealed_stacked_gate_result(
            name=_BATTERY_GATE_NAME,
            passed=False,
            failures=tuple(registration_failures),
            details={
                **coverage_details,
                "registered_specs": len(specs),
                "tested_comparisons": 0,
                "untestable_comparisons": [],
                "comparisons": {},
                **(
                    {_TAIL_SUPPORT_GATE_DETAIL_KEY: tail_support_receipt}
                    if tail_support_receipt is not None
                    else {}
                ),
            },
        )
    validate_stacked_spine_frame(frame, boundary="by-origin battery")

    failures: list[str] = []
    comparisons: dict[str, object] = {}
    structural_absence_receipts: dict[str, dict[str, object]] = {}
    untestable: list[str] = []
    tested = 0
    declared_absence_rules: dict[
        tuple[str, str], tuple[GapFillDirection, GapFillAbsenceRule]
    ] = {}
    for direction in (*authority.gap_fill_plan, *_canonical_gap_fill_plan):
        for rule in direction.recipient_absence_rules:
            declared_absence_rules[(rule.entity, rule.column)] = (direction, rule)
    for spec in specs:
        table = frame.table(spec.entity)
        channel = table[support_channel_column(spec.entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(spec.entity)],
            errors="raise",
        ).astype("int64")
        weights = np.asarray(
            frame.resolve_weights(spec.entity).values,
            dtype=np.float64,
        )
        scope = (clone_index.eq(spec.clone_index)).to_numpy() & (weights > 0.0)
        for column, metric in spec.column_metrics.items():
            label = f"{spec.entity}/{spec.family}/{column}[clone_{spec.clone_index}]"
            if column not in table.columns:
                failures.append(f"{label}: registered column is absent from the frame.")
                comparisons[label] = {
                    "status": "missing_column",
                    "metric": metric,
                }
                continue
            series = table[column]
            target_scope = scope.copy()
            structural_rule = declared_absence_rules.get((spec.entity, column))
            if structural_rule is not None:
                rule_direction, rule = structural_rule
                structural_mask, structural_receipt = _gap_fill_absence_rule_mask(
                    frame,
                    direction=rule_direction,
                    rule=rule,
                )
                scoped_structural = structural_mask.to_numpy(dtype=bool) & scope
                scoped_null_mask = series.isna().to_numpy(dtype=bool) & scope
                unexpected = int((scoped_null_mask & ~scoped_structural).sum())
                synthesized = int((scoped_structural & ~scoped_null_mask).sum())
                structural_receipt = {
                    **structural_receipt,
                    "comparison_clone_index": spec.clone_index,
                    "rows_excluded_from_scope": int(scoped_structural.sum()),
                    "unexpected_null_rows": unexpected,
                    "structural_rows_filled": synthesized,
                }
                structural_absence_receipts[label] = structural_receipt
                if unexpected or synthesized:
                    failures.append(
                        f"{label}: exact structural-absence equation failed; "
                        f"unexpected_null_rows={unexpected}, "
                        f"structural_rows_filled={synthesized}."
                    )
                    comparisons[label] = {
                        "status": "structural_absence_mismatch",
                        "metric": metric,
                    }
                    continue
                target_scope &= ~scoped_structural
            left_rows = target_scope & channel.eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
            right_rows = (
                target_scope & channel.eq(ACS_STACKED_SUPPORT_CHANNEL).to_numpy()
            )
            scoped_nulls = int(series.isna().to_numpy(dtype=bool)[target_scope].sum())
            if scoped_nulls:
                failures.append(
                    f"{label}: {scoped_nulls} null value(s) inside the "
                    "comparison scope; the battery runs only on completed "
                    "surfaces."
                )
                comparisons[label] = {
                    "status": "null_in_scope",
                    "metric": metric,
                    "null_rows": scoped_nulls,
                }
                continue
            invalid_mask, invalidity = _declared_metric_invalidity(
                series,
                metric=metric,
                scope=target_scope,
            )
            invalid_rows = int(invalid_mask.sum())
            if invalid_rows:
                invalid_counts = _nonzero_invalidity_counts(invalidity)
                failures.append(
                    f"{label}: declared {metric} metric has {invalid_rows} "
                    "invalid value(s) inside the comparison scope, including "
                    f"{invalid_counts}."
                )
                comparisons[label] = {
                    "status": "invalid_values",
                    "metric": metric,
                    "invalid_rows": invalid_rows,
                    "invalidity": invalid_counts,
                }
                continue
            values: np.ndarray | None = None
            if metric != "categorical_tvd":
                values = _battery_numeric_values(
                    label,
                    series,
                    metric=metric,
                    scope=target_scope,
                    failures=failures,
                )
                if values is None:
                    comparisons[label] = {
                        "status": "invalid_values",
                        "metric": metric,
                    }
                    continue
            support = {
                "asec": int(left_rows.sum()),
                "acs": int(right_rows.sum()),
            }
            if min(support.values()) < authority.support_profile.min_effective_support:
                comparisons[label] = {
                    "status": "insufficient_support",
                    "metric": metric,
                    "scope_rows": support,
                }
                untestable.append(label)
                continue
            tested += 1
            if metric == "categorical_tvd":
                _battery_categorical_comparison(
                    label=label,
                    series=series,
                    left_rows=left_rows,
                    right_rows=right_rows,
                    weights=weights,
                    failures=failures,
                    comparisons=comparisons,
                )
                continue
            if values is None:  # pragma: no cover - categorical continued above
                raise AssertionError("Numeric battery metric has no values.")
            if metric in {"boolean_incidence", "rare_incidence"}:
                _battery_incidence_comparison(
                    label=label,
                    metric=metric,
                    values=values,
                    left_rows=left_rows,
                    right_rows=right_rows,
                    weights=weights,
                    failures=failures,
                    comparisons=comparisons,
                )
            else:
                _battery_sign_separated_comparison(
                    label=label,
                    values=values,
                    left_rows=left_rows,
                    right_rows=right_rows,
                    weights=weights,
                    failures=failures,
                    comparisons=comparisons,
                    min_effective_support=(
                        authority.support_profile.min_effective_support
                    ),
                )
    for target, metric in authority.joint_metric_registry.items():
        entity, family, columns, clone_role = target
        label = _joint_battery_target_label(target)
        table = frame.table(entity)
        missing_columns = sorted(set(columns) - set(table.columns))
        if missing_columns:
            failures.append(
                f"{label}: registered column(s) are absent from the frame: "
                f"{missing_columns}."
            )
            comparisons[label] = {
                "status": "missing_column",
                "metric": metric,
                "missing_columns": missing_columns,
            }
            continue
        channel = table[support_channel_column(entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(entity)],
            errors="raise",
        ).astype("int64")
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        scope = clone_index.eq(clone_role).to_numpy() & (weights > 0.0)
        left_rows = scope & channel.eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
        right_rows = scope & channel.eq(ACS_STACKED_SUPPORT_CHANNEL).to_numpy()
        scoped_nulls = int(table.loc[:, list(columns)].isna().any(axis=1)[scope].sum())
        if scoped_nulls:
            failures.append(
                f"{label}: {scoped_nulls} null tuple(s) inside the comparison "
                "scope; the battery runs only on completed surfaces."
            )
            comparisons[label] = {
                "status": "null_in_scope",
                "metric": metric,
                "null_rows": scoped_nulls,
            }
            continue
        support = {"asec": int(left_rows.sum()), "acs": int(right_rows.sum())}
        if min(support.values()) < authority.support_profile.min_effective_support:
            comparisons[label] = {
                "status": "insufficient_support",
                "metric": metric,
                "scope_rows": support,
            }
            untestable.append(label)
            continue
        tested += 1
        tuples = pd.Series(
            list(table.loc[:, list(columns)].itertuples(index=False, name=None)),
            index=table.index,
            dtype=object,
        )
        _battery_categorical_comparison(
            label=label,
            series=tuples,
            left_rows=left_rows,
            right_rows=right_rows,
            weights=weights,
            failures=failures,
            comparisons=comparisons,
        )
    for label, receipt in structural_absence_receipts.items():
        comparison = comparisons.get(label)
        if isinstance(comparison, dict):
            comparison["recipient_absence_authority"] = receipt
    return _sealed_stacked_gate_result(
        name=_BATTERY_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={
            **coverage_details,
            "tolerances": {
                "incidence_ratio_bounds": list(_BATTERY_INCIDENCE_RATIO_BOUNDS),
                "max_quantile_envelope_distance": (
                    _BATTERY_QUANTILE_ENVELOPE_TOLERANCE
                ),
                "max_categorical_total_variation_distance": (
                    _BATTERY_CATEGORICAL_TVD_TOLERANCE
                ),
            },
            "registered_specs": len(specs),
            "tested_comparisons": tested,
            "untestable_comparisons": sorted(untestable),
            "comparisons": comparisons,
            **(
                {_TAIL_SUPPORT_GATE_DETAIL_KEY: tail_support_receipt}
                if tail_support_receipt is not None
                else {}
            ),
        },
    )


def _battery_target_label(target: tuple[str, str, str, int]) -> str:
    entity, family, column, clone_index = target
    return f"{entity}/{family}/{column}[clone_{clone_index}]"


def _joint_battery_target_label(
    target: tuple[str, str, tuple[str, ...], int],
) -> str:
    entity, family, columns, clone_index = target
    return f"{entity}/{family}/joint[{','.join(columns)}][clone_{clone_index}]"


def _battery_numeric_values(
    label: str,
    series: pd.Series,
    *,
    metric: str,
    scope: np.ndarray,
    failures: list[str],
) -> np.ndarray | None:
    values = np.zeros(len(series), dtype=np.float64)
    try:
        values[scope] = pd.to_numeric(
            series.iloc[np.flatnonzero(scope)],
            errors="raise",
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        failures.append(f"{label}: declared {metric} metric requires numeric values.")
        return None
    non_finite = ~np.isfinite(values[scope])
    if non_finite.any():
        failures.append(
            f"{label}: declared {metric} metric requires finite numeric values; "
            f"found {int(non_finite.sum())} non-finite value(s)."
        )
        return None
    if metric == "boolean_incidence":
        invalid = ~np.isin(values[scope], (0.0, 1.0))
        if invalid.any():
            failures.append(
                f"{label}: declared boolean incidence requires values in "
                f"{{0, 1}}; found {int(invalid.sum())} other value(s)."
            )
            return None
    return values


def _battery_incidence_comparison(
    *,
    label: str,
    metric: str,
    values: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    weights: np.ndarray,
    failures: list[str],
    comparisons: dict[str, object],
) -> None:
    left = _weighted_nonzero_incidence(values[left_rows], weights[left_rows])
    right = _weighted_nonzero_incidence(values[right_rows], weights[right_rows])
    record: dict[str, object] = {
        "status": "tested",
        "metric": metric,
        "asec_incidence": left,
        "acs_incidence": right,
        "nonzero_rows": {
            "asec": int((values[left_rows] != 0.0).sum()),
            "acs": int((values[right_rows] != 0.0).sum()),
        },
    }
    comparisons[label] = record
    if left == 0.0 and right == 0.0:
        failures.append(
            f"{label}: zero weighted incidence on both origins with adequate "
            "support; the registered comparison is dead."
        )
        record["status"] = "dead_comparison"
        return
    ratio = math.inf if left == 0.0 else right / left
    record["incidence_ratio_acs_over_asec"] = ratio if math.isfinite(ratio) else "inf"
    lower, upper = _BATTERY_INCIDENCE_RATIO_BOUNDS
    if not lower <= ratio <= upper:
        failures.append(
            f"{label}: weighted incidence ratio {ratio:.6g} is outside "
            f"[{lower:.6g}, {upper:.6g}] (asec={left:.6g}, acs={right:.6g})."
        )


def _battery_sign_separated_comparison(
    *,
    label: str,
    values: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    weights: np.ndarray,
    failures: list[str],
    comparisons: dict[str, object],
    min_effective_support: int,
) -> None:
    record: dict[str, object] = {
        "status": "tested",
        "metric": "monetary_sign_separated",
        "legs": {},
    }
    comparisons[label] = record
    lower, upper = _BATTERY_INCIDENCE_RATIO_BOUNDS
    for leg_name, leg_mask in (
        ("positive", values > 0.0),
        ("negative", values < 0.0),
    ):
        left_leg = leg_mask & left_rows
        right_leg = leg_mask & right_rows
        left_incidence = _weighted_mask_incidence(
            left_leg[left_rows], weights[left_rows]
        )
        right_incidence = _weighted_mask_incidence(
            right_leg[right_rows], weights[right_rows]
        )
        leg_record: dict[str, object] = {
            "asec_incidence": left_incidence,
            "acs_incidence": right_incidence,
            "nonzero_rows": {
                "asec": int(left_leg.sum()),
                "acs": int(right_leg.sum()),
            },
        }
        record["legs"][leg_name] = leg_record
        if left_incidence == 0.0 and right_incidence == 0.0:
            leg_record["status"] = "absent_on_both_origins"
            continue
        ratio = math.inf if left_incidence == 0.0 else right_incidence / left_incidence
        leg_record["incidence_ratio_acs_over_asec"] = (
            ratio if math.isfinite(ratio) else "inf"
        )
        if not lower <= ratio <= upper:
            failures.append(
                f"{label}/{leg_name}: weighted {leg_name}-leg incidence ratio "
                f"{ratio:.6g} is outside [{lower:.6g}, {upper:.6g}] "
                f"(asec={left_incidence:.6g}, acs={right_incidence:.6g})."
            )
        if (
            int(left_leg.sum()) < min_effective_support
            or int(right_leg.sum()) < min_effective_support
        ):
            leg_record["quantile_envelope"] = "leg_insufficient_support"
            continue
        left_quantiles = _battery_conditional_quantiles(
            np.abs(values[left_leg]), weights[left_leg]
        )
        right_quantiles = _battery_conditional_quantiles(
            np.abs(values[right_leg]), weights[right_leg]
        )
        distance = _battery_quantile_envelope_distance(left_quantiles, right_quantiles)
        leg_record["quantile_envelope_distance"] = distance
        if distance > _BATTERY_QUANTILE_ENVELOPE_TOLERANCE:
            failures.append(
                f"{label}/{leg_name}: conditional-quantile envelope distance "
                f"{distance:.6g} exceeds "
                f"{_BATTERY_QUANTILE_ENVELOPE_TOLERANCE:.6g}."
            )


def _battery_categorical_comparison(
    *,
    label: str,
    series: pd.Series,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    weights: np.ndarray,
    failures: list[str],
    comparisons: dict[str, object],
) -> None:
    values = series.to_numpy(dtype=object)
    distributions: dict[str, dict[str, float]] = {}
    for origin, rows in (("asec", left_rows), ("acs", right_rows)):
        origin_weights = weights[rows]
        total = float(origin_weights.sum())
        shares: dict[str, float] = {}
        for value, weight in zip(values[rows], origin_weights, strict=True):
            shares[str(value)] = shares.get(str(value), 0.0) + float(weight)
        distributions[origin] = {
            category: share / total for category, share in shares.items()
        }
    categories = sorted(set(distributions["asec"]) | set(distributions["acs"]))
    distance = 0.5 * sum(
        abs(
            distributions["asec"].get(category, 0.0)
            - distributions["acs"].get(category, 0.0)
        )
        for category in categories
    )
    comparisons[label] = {
        "status": "tested",
        "metric": "categorical_tvd",
        "total_variation_distance": distance,
        "category_shares": distributions,
    }
    if distance > _BATTERY_CATEGORICAL_TVD_TOLERANCE:
        failures.append(
            f"{label}: categorical total-variation distance {distance:.6g} "
            f"exceeds {_BATTERY_CATEGORICAL_TVD_TOLERANCE:.6g}."
        )


def _weighted_nonzero_incidence(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[values != 0.0].sum() / total)


def _weighted_mask_incidence(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[mask].sum() / total)


def _battery_conditional_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    positions = np.minimum(
        np.searchsorted(cumulative, np.asarray(_BATTERY_QUANTILES), side="left"),
        len(sorted_values) - 1,
    )
    return sorted_values[positions]


def _battery_quantile_envelope_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Battery quantile envelopes require finite values.")
    denominator = np.abs(left) + np.abs(right)
    distances = np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    distance = float(np.max(distances))
    if not math.isfinite(distance):
        raise ValueError("Battery quantile-envelope distance must be finite.")
    return distance
