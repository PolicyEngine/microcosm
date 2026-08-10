"""Deterministic, mass-conserving transfer of the PUF capital-gains tail."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult, tail_concentration_gate
from microcosm.build.us_runtime.puf_aggregate_records import (
    PufAggregateDisaggregationSpec,
    load_default_puf_aggregate_disaggregation_spec,
)
from microcosm.build.us_runtime.puf_interest_components import (
    US_PUF_E19200_AGI_BANDS,
    puf_e19200_interest_components_asset_identity,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.build.us_runtime.support_provenance import (
    PUF_TAX_DETAIL_CLONE_INDEX,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    puf_tax_detail_clone_mask,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, Weights

__all__ = [
    "PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION",
    "PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MAX_TOP_SHARE",
    "PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS",
    "PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_TOP_K",
    "PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS",
    "PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET",
    "PUF_CAPITAL_GAINS_TAIL_QUANTILE",
    "PUF_CAPITAL_GAINS_TAIL_SUPPORT_CONTRACT_VERSION",
    "PUF_CAPITAL_GAINS_TAIL_STAGE_NAME",
    "PUF_CAPITAL_GAINS_TAIL_SUPPORT_CHANNEL",
    "PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS",
    "PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN",
    "assert_puf_capital_gains_tail_survives_selection",
    "puf_capital_gains_tail_concentration_gate",
    "puf_capital_gains_tail_concentration_controls_identity",
    "puf_capital_gains_tail_execution_inputs_identity",
    "puf_capital_gains_tail_spec_identity",
    "puf_capital_gains_tail_support_contract_identity",
    "puf_capital_gains_tail_terminal_support_receipt",
    "select_puf_capital_gains_tail_donors",
    "transfer_puf_capital_gains_tail",
    "validate_puf_capital_gains_tail_manifest",
    "validate_puf_capital_gains_tail_terminal_support_receipt",
    "write_puf_capital_gains_tail_manifest",
]

PUF_CAPITAL_GAINS_TAIL_STAGE_NAME = "capital_gains_tail_transfer"
PUF_CAPITAL_GAINS_TAIL_SUPPORT_CHANNEL = PUF_TAX_DETAIL_SUPPORT_CHANNEL
PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION = 2
PUF_CAPITAL_GAINS_TAIL_SUPPORT_CONTRACT_VERSION = 1
PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET = 1_270_900_000_000.0
PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_TOP_K = 100
PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MAX_TOP_SHARE = 0.75
PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS = 500

# microcosm#567 diagnostic geometry: recipient predictors are bounded by the
# $1,999,998 ASEC capital-gains topcode. Weighted q99.5 of positive donor
# ST+LT is $1,685,506.66, while the next tested upper-tail boundary is already
# above the recipient ceiling. This is a declared source stratum boundary,
# not a calibration knob or target multiplier.
PUF_CAPITAL_GAINS_TAIL_QUANTILE = 0.995
_NEXT_REFERENCE_QUANTILE = 0.999
_ASEC_CAPITAL_GAINS_TOPCODE = 1_999_998.0

PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS = (
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "long_term_capital_gains_on_collectibles",
    "non_sch_d_capital_gains",
)
PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS = ("unrecaptured_section_1250_gain",)
_JOINT_VECTOR_COLUMNS = (
    *PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    *PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
)
#: Donor/candidate schema overlap with declared ownership (microcosm#570
#: review hardening): the recipient candidate row is the full tax-unit
#: table row, so every donor tax-unit-grain OUTPUT column collides. The
#: joint CG vector is donor-owned (re-overlaid after the merge); the
#: remaining donor tax-unit outputs (mortgage/ALD family) are
#: recipient-owned — the clone keeps its household's own values, only
#: capital gains transfer. A new collision outside this partition fails
#: loud instead of silently replacing donor payload.
_RECIPIENT_OWNED_CANDIDATE_OVERLAP = frozenset(
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
) - set(PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS)

_COMBINED_COLUMN = "short_term_plus_long_term_capital_gains"
_TAIL_COMBINED_COLUMN = "_puf_capital_gains_tail_combined"
_TAIL_AGI_BAND_INDEX_COLUMN = "_puf_capital_gains_tail_agi_band_index"
_TAIL_AGI_BAND_LABEL_COLUMN = "_puf_capital_gains_tail_agi_band_label"
_TAIL_SYNTHETIC_COLUMN = "_puf_capital_gains_tail_is_synthetic"

PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN = "puf_capital_gains_tail_transfer_applied"
PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN = "puf_capital_gains_tail_donor_source_id"
PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN = (
    "puf_capital_gains_tail_donor_is_synthetic"
)
PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN = (
    "puf_capital_gains_tail_donor_filing_status_code"
)
PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN = (
    "puf_capital_gains_tail_donor_agi_band_index"
)
PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN = "puf_capital_gains_tail_transfer_weight"

_FILING_STATUS_BY_CODE = {
    1: "SINGLE",
    2: "JOINT",
    3: "SEPARATE",
    4: "HEAD_OF_HOUSEHOLD",
    5: "SURVIVING_SPOUSE",
}
_FILING_STATUS_CODE = {value: key for key, value in _FILING_STATUS_BY_CODE.items()}
_RECIPIENT_AGI_PROXY_COLUMNS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
)
_AGI_UPPER_BOUNDS = np.asarray(
    [
        band.upper_bound
        for band in US_PUF_E19200_AGI_BANDS
        if band.upper_bound is not None
    ],
    dtype=np.float64,
)


def puf_capital_gains_tail_support_contract_identity() -> dict[str, object]:
    """Return the immutable per-filing-status recipient-support doctrine."""

    return {
        "contract_id": "puf_capital_gains_tail_per_filing_status_support",
        "version": PUF_CAPITAL_GAINS_TAIL_SUPPORT_CONTRACT_VERSION,
        "partition": "filing_status",
        "filing_statuses": [
            {"filing_status_code": code, "filing_status": name}
            for code, name in _FILING_STATUS_BY_CODE.items()
        ],
        "candidate_universe": (
            "unique single-tax-unit PUF-detail recipient households with "
            "weight capacity for the global maximum assigned tail-donor weight"
        ),
        "required_minimum": "selected_q99_5_tail_donor_count_in_filing_status",
        "insufficient_support_action": (
            "skip_entire_filing_status_attachment_without_widening"
        ),
        "agi_band_policy": (
            "nearest_band_first_then_all_agi_bands_within_filing_status"
        ),
        "agi_band_count": len(US_PUF_E19200_AGI_BANDS),
    }


def puf_capital_gains_tail_spec_identity(
    spec: PufAggregateDisaggregationSpec | None = None,
) -> dict[str, object]:
    """Return the exact resolved aggregate-disaggregation input to tail selection."""

    resolved = spec or load_default_puf_aggregate_disaggregation_spec()
    resolved.validate()
    return {
        "enabled": resolved.enabled,
        "forbes_top_tail": resolved.forbes_top_tail,
        "source": resolved.source,
        "aggregate_recids": list(resolved.aggregate_recids),
        "synthetic_recid_start": resolved.synthetic_recid_start,
        "screened_fields": list(resolved.screened_fields),
        "synthetic_tail_support_eligible": (resolved.synthetic_tail_support_eligible),
        "buckets": [
            {
                "recid": recid,
                "description": bucket.description,
                "agi_lower": bucket.agi_lower,
                "agi_upper": bucket.agi_upper,
                "synthetic_agi_upper": bucket.synthetic_agi_upper,
            }
            for recid, bucket in sorted(resolved.buckets.items())
        ],
    }


def puf_capital_gains_tail_concentration_controls_identity() -> dict[str, object]:
    """Return the explicit selected-tail and produced-frame gate controls."""

    return {
        "top_k": PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_TOP_K,
        "max_top_share": PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MAX_TOP_SHARE,
        "min_nonzero_records": (
            PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS
        ),
        "reviewed_exclusions": {},
    }


def puf_capital_gains_tail_execution_inputs_identity() -> dict[str, object]:
    """Bind the data assets and controls read by the tail callback."""

    return {
        "aggregate_disaggregation_spec": puf_capital_gains_tail_spec_identity(),
        "soi_e19200_agi_bands": (puf_e19200_interest_components_asset_identity()),
        "concentration_gate": (
            puf_capital_gains_tail_concentration_controls_identity()
        ),
    }


def select_puf_capital_gains_tail_donors(
    donor: pd.DataFrame,
    *,
    spec: PufAggregateDisaggregationSpec | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the declared weighted-positive ST+LT tail stratum."""

    required = {
        "tax_unit_id",
        "weight",
        "filing_status_code",
        PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN,
        *_JOINT_VECTOR_COLUMNS,
    }
    missing = sorted(required - set(donor.columns))
    if missing:
        raise ValueError(f"PUF capital-gains tail donor missing columns: {missing}.")
    resolved_spec = spec or load_default_puf_aggregate_disaggregation_spec()
    resolved_spec.validate()

    numeric = donor.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="raise")
        values = numeric[column].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(
                f"PUF capital-gains tail donor column {column!r} must be finite."
            )
    weights = numeric["weight"].to_numpy(dtype=np.float64, copy=False)
    if (weights < 0.0).any() or not (weights > 0.0).any():
        raise ValueError("PUF capital-gains tail donor weights must be nonnegative.")
    source_ids = numeric["tax_unit_id"].to_numpy(dtype=np.int64, copy=False)
    if pd.Series(source_ids).duplicated().any():
        raise ValueError("PUF capital-gains tail donor source IDs must be unique.")

    combined = numeric["short_term_capital_gains"].to_numpy(
        dtype=np.float64, copy=False
    ) + numeric["long_term_capital_gains_before_response"].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    synthetic = source_ids >= resolved_spec.synthetic_recid_start
    eligible = (
        np.ones(len(numeric), dtype=bool)
        if resolved_spec.synthetic_tail_support_eligible
        else ~synthetic
    )
    positive = eligible & (weights > 0.0) & (combined > 0.0)
    if not positive.any():
        raise ValueError("PUF capital-gains tail has no eligible positive ST+LT rows.")
    boundary = _weighted_quantile(
        combined[positive],
        weights[positive],
        PUF_CAPITAL_GAINS_TAIL_QUANTILE,
    )
    next_reference_boundary = _weighted_quantile(
        combined[positive],
        weights[positive],
        _NEXT_REFERENCE_QUANTILE,
    )
    if boundary > _ASEC_CAPITAL_GAINS_TOPCODE:
        raise ValueError(
            "PUF capital-gains q99.5 boundary exceeds the measured ASEC "
            f"recipient topcode: {boundary} > {_ASEC_CAPITAL_GAINS_TOPCODE}."
        )
    if next_reference_boundary <= _ASEC_CAPITAL_GAINS_TOPCODE:
        raise ValueError(
            "PUF capital-gains diagnostic geometry changed: weighted-positive "
            f"q{_NEXT_REFERENCE_QUANTILE}={next_reference_boundary} no longer "
            f"exceeds the ASEC recipient topcode {_ASEC_CAPITAL_GAINS_TOPCODE}."
        )

    # The stratum is strictly above the declared boundary. This follows the
    # issue's "above q99.5" contract and avoids transferring a point mass tied
    # at the quantile observation itself.
    tail_mask = positive & (combined > boundary)
    if not tail_mask.any():
        raise ValueError(
            "PUF capital-gains donor has no records strictly above its q99.5 boundary."
        )
    tail = numeric.loc[tail_mask].copy()
    tail[_TAIL_COMBINED_COLUMN] = combined[tail_mask]
    tail[_TAIL_SYNTHETIC_COLUMN] = synthetic[tail_mask]
    band_index = _agi_band_indices(
        tail[PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN].to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )
    tail[_TAIL_AGI_BAND_INDEX_COLUMN] = band_index
    tail[_TAIL_AGI_BAND_LABEL_COLUMN] = [
        US_PUF_E19200_AGI_BANDS[index].label for index in band_index
    ]
    tail.sort_values("tax_unit_id", kind="mergesort", inplace=True)
    tail.reset_index(drop=True, inplace=True)

    eligible_positive_mass = float(
        np.dot(np.maximum(combined[positive], 0.0), weights[positive])
    )
    tail_positive_mass = float(
        np.dot(
            tail[_TAIL_COMBINED_COLUMN].to_numpy(dtype=np.float64),
            tail["weight"].to_numpy(dtype=np.float64),
        )
    )
    receipt: dict[str, object] = {
        "quantile": PUF_CAPITAL_GAINS_TAIL_QUANTILE,
        "comparison": "strictly_greater_than",
        "realized_boundary": float(boundary),
        "next_reference_quantile": _NEXT_REFERENCE_QUANTILE,
        "next_reference_boundary": float(next_reference_boundary),
        "recipient_topcode": _ASEC_CAPITAL_GAINS_TOPCODE,
        "eligible_positive_record_count": int(positive.sum()),
        "eligible_positive_weight": float(weights[positive].sum()),
        "eligible_positive_mass": eligible_positive_mass,
        "tail_record_count": int(len(tail)),
        "tail_weight": float(tail["weight"].sum()),
        "tail_positive_mass": tail_positive_mass,
        "tail_positive_mass_share": float(tail_positive_mass / eligible_positive_mass),
        "synthetic_tail_support_eligible": bool(
            resolved_spec.synthetic_tail_support_eligible
        ),
        "synthetic_recid_start": int(resolved_spec.synthetic_recid_start),
        "synthetic_tail_record_count": int(
            tail[_TAIL_SYNTHETIC_COLUMN].to_numpy(dtype=bool).sum()
        ),
    }
    return tail, receipt


def puf_capital_gains_tail_concentration_gate(
    tail: pd.DataFrame,
    *,
    weights: Sequence[float] | None = None,
) -> GateResult:
    """Run the existing #462 weighted top-100 diagnostic on a tail stratum."""

    missing = sorted(set(_JOINT_VECTOR_COLUMNS) - set(tail.columns))
    if missing:
        raise ValueError(f"PUF capital-gains tail missing vector columns: {missing}.")
    resolved_weights = (
        tail["weight"].to_numpy(dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    values = {
        column: tail[column].to_numpy(dtype=np.float64)
        for column in _JOINT_VECTOR_COLUMNS
    }
    values[_COMBINED_COLUMN] = (
        values["short_term_capital_gains"]
        + values["long_term_capital_gains_before_response"]
    )
    return tail_concentration_gate(
        values,
        {column: resolved_weights for column in values},
        top_k=PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_TOP_K,
        max_top_share=PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MAX_TOP_SHARE,
        min_nonzero_records=(PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS),
        reviewed_exclusions={},
    )


def assert_puf_capital_gains_tail_survives_selection(
    base_frame: Frame,
    selected_frame: Frame,
    *,
    require_present: bool = False,
) -> dict[str, object]:
    """Require frozen-support recovery to retain every materialized tail donor."""

    base_tax_unit = base_frame.table("tax_unit")
    if PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN not in base_tax_unit:
        if require_present:
            raise ValueError(
                "PUF capital-gains own-tail provenance is absent from the base. "
                "Rebuild the base through capital_gains_tail_transfer before "
                "release."
            )
        return {
            "passed": True,
            "status": "tail_not_present",
            "base_tail_record_count": 0,
            "selected_tail_record_count": 0,
            "missing_tail_record_count": 0,
        }
    selected_tax_unit = selected_frame.table("tax_unit")
    required = {
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    }
    missing = sorted(required - set(selected_tax_unit.columns))
    if missing:
        raise ValueError(
            "Frozen support dropped PUF capital-gains tail provenance columns: "
            f"{missing}."
        )

    def tail_rows(frame: Frame) -> pd.DataFrame:
        tax_unit = frame.table("tax_unit")
        applied = (
            tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].astype(bool).to_numpy()
        )
        rows = pd.DataFrame(
            {
                "donor_source_id": tax_unit.loc[
                    applied,
                    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
                ].to_numpy(dtype=np.int64),
                "effective_weight": frame.resolve_weights("tax_unit").values[applied],
            }
        )
        if rows["donor_source_id"].duplicated().any():
            raise ValueError(
                "PUF capital-gains tail donor provenance must remain one-to-one."
            )
        return rows.sort_values("donor_source_id", kind="mergesort").reset_index(
            drop=True
        )

    base_tail = tail_rows(base_frame)
    selected_tail = tail_rows(selected_frame)
    missing_ids = sorted(
        set(base_tail["donor_source_id"]) - set(selected_tail["donor_source_id"])
    )
    extra_ids = sorted(
        set(selected_tail["donor_source_id"]) - set(base_tail["donor_source_id"])
    )
    if missing_ids or extra_ids:
        raise ValueError(
            "Frozen support does not preserve the PUF capital-gains own-tail "
            f"stratum: missing {len(missing_ids)} donor(s), extra "
            f"{len(extra_ids)} donor(s). Regenerate the selection-source "
            "manifest from the rebuilt support before release."
        )
    if not np.array_equal(
        base_tail["effective_weight"].to_numpy(dtype=np.float64),
        selected_tail["effective_weight"].to_numpy(dtype=np.float64),
    ):
        raise ValueError(
            "Frozen support changed PUF capital-gains tail design weights before "
            "calibration."
        )
    donor_ids = base_tail["donor_source_id"].astype("int64").tolist()
    return {
        "passed": True,
        "status": "retained",
        "base_tail_record_count": int(len(base_tail)),
        "selected_tail_record_count": int(len(selected_tail)),
        "missing_tail_record_count": 0,
        "tail_design_weight": float(base_tail["effective_weight"].sum()),
        "donor_source_ids_sha256": _canonical_sha256(donor_ids),
    }


#: Float tolerance for the worsening comparator (microcosm#571 review):
#: weight splitting and changed summation order move shares by ULPs; a
#: strict > would fail 0.84 -> 0.8400000000000001. Anything below this is
#: numerical noise, not a worsening.
_WORSENING_SHARE_TOLERANCE = 1e-9


def _raw_top_share_receipts(
    values_by_column: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    top_k: int = PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_TOP_K,
) -> dict[str, dict[str, object]]:
    """Measure every column's weighted top-share raw — no thin-column skip.

    microcosm#571 review, Critical: the shared gate omits shares for columns
    under its min-carriers floor, so a thin PRE-stage column (collectibles:
    97 carriers, literal top-100 share 100%) read as pre_share 0.0 and the
    stage's improvement (100% -> 83.9%, 97 -> 1,135 carriers) was reported
    as a worsening from zero. The comparator must see the raw geometry.
    """

    weight_vector = np.asarray(weights, dtype=np.float64)
    receipts: dict[str, dict[str, object]] = {}
    for column, values in values_by_column.items():
        value_vector = np.asarray(values, dtype=np.float64)
        # The production gate's population: finite rows with positive
        # weighted |mass| (microcosm#571 round 2 — a single NaN otherwise
        # poisons the raw share to 0.0 while the gate reads 0.95/FAIL,
        # making pre/post incommensurable with over-threshold membership).
        # Only the gate's minimum-carrier floor is deliberately omitted.
        finite = np.isfinite(value_vector) & np.isfinite(weight_vector)
        masked_mass = np.abs(value_vector[finite]) * weight_vector[finite]
        positive = masked_mass > 0.0
        masked_mass = masked_mass[positive]
        carriers = int(masked_mass.size)
        total = float(masked_mass.sum())
        if total > 0.0:
            top = np.sort(masked_mass)[::-1][:top_k]
            share = float(top.sum() / total)
        else:
            share = 0.0
        carrier_values = value_vector[finite][positive]
        receipts[column] = {
            "top_share": share,
            "carriers": carriers,
            "distinct_values": int(np.unique(carrier_values).size),
        }
    return receipts


def _stage_attributable_concentration_failures(
    pre_receipts: Mapping[str, Mapping[str, object]],
    post_receipts: Mapping[str, Mapping[str, object]],
    post_gate: GateResult,
) -> tuple[list[str], dict[str, dict[str, object]]]:
    """Fail only columns the stage left over threshold AND strictly worsened.

    microcosm#567/#570 adjudication: the stage's frame check exists to catch
    stage-caused pathology (a broadcast or a bad transfer concentrating
    mass). Concentration the frame already carried BEFORE the stage is
    owned by the release-side gate, which measures the calibrated artifact
    and has its own reviewed per-run register. Over-threshold membership
    comes from the production gate on the post frame; the pre/post shares
    come from the raw measurement so thin pre-stage columns compare
    truthfully. Per-column receipts (shares, carriers, distinct values —
    the last a visibility signal for share-preserving value collapses)
    ride the stage manifest either way.
    """

    over_threshold = {line.partition(":")[0].strip() for line in post_gate.failures}
    failures: list[str] = []
    receipts: dict[str, dict[str, object]] = {}
    for column in sorted(post_receipts):
        pre = pre_receipts.get(column, {})
        pre_share = float(pre.get("top_share", 0.0))
        post_share = float(post_receipts[column]["top_share"])
        over = column in over_threshold
        worsened = post_share > pre_share + _WORSENING_SHARE_TOLERANCE
        receipts[column] = {
            "pre_stage_top_share": pre_share,
            "post_stage_top_share": post_share,
            "pre_stage_carriers": int(pre.get("carriers", 0)),
            "post_stage_carriers": int(post_receipts[column]["carriers"]),
            "pre_stage_distinct_values": int(pre.get("distinct_values", 0)),
            "post_stage_distinct_values": int(post_receipts[column]["distinct_values"]),
            "over_threshold": over,
            "stage_worsened_share": worsened,
        }
        if over and worsened:
            failures.append(
                f"{column}: post-stage top-100 share {post_share:.4f} is over "
                f"the threshold and worsened from pre-stage {pre_share:.4f} — "
                "the transfer caused or deepened the concentration."
            )
    return failures, receipts


def transfer_puf_capital_gains_tail(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    spec: PufAggregateDisaggregationSpec | None = None,
) -> tuple[Frame, dict[str, object]]:
    """Split PUF households and transfer exact joint donor-tail vectors."""

    if frame.schema != US_SCHEMA:
        raise ValueError("PUF capital-gains tail transfer requires the US schema.")
    if frame.links:
        raise ValueError(
            "PUF capital-gains tail transfer does not support link tables."
        )
    if frame.weighted_entities != ("household",):
        raise ValueError(
            "PUF capital-gains tail transfer requires household to be the sole "
            f"explicitly weighted entity, got {frame.weighted_entities}."
        )
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("PUF capital-gains tail seed must be a nonnegative integer.")

    resolved_spec = spec or load_default_puf_aggregate_disaggregation_spec()
    selected_tail, selection = select_puf_capital_gains_tail_donors(
        donor,
        spec=resolved_spec,
    )
    donor_weight_total = float(pd.to_numeric(donor["weight"], errors="raise").sum())
    if not np.isfinite(donor_weight_total) or donor_weight_total <= 0.0:
        raise ValueError("PUF donor total weight must be positive and finite.")
    frame_household_weight_total = frame.weights_for("household").total
    design_weight_normalization = frame_household_weight_total / donor_weight_total
    selected_assigned_weights = (
        selected_tail["weight"].to_numpy(dtype=np.float64) * design_weight_normalization
    )
    if not (selected_assigned_weights > 0.0).all():
        raise ValueError("Every selected PUF tail donor must receive positive mass.")

    concentration = puf_capital_gains_tail_concentration_gate(
        selected_tail,
        weights=selected_assigned_weights,
    )
    if not concentration.passed:
        raise ValueError(
            "PUF capital-gains tail fails the existing weighted concentration "
            "gate:\n  " + "\n  ".join(concentration.failures)
        )

    candidates = _recipient_candidates(
        frame,
        maximum_transfer_weight=float(selected_assigned_weights.max()),
        seed=int(seed),
    )
    recipient_support = _recipient_support_receipt(selected_tail, candidates)
    attached_codes = {
        int(stratum["filing_status_code"])
        for stratum in recipient_support["strata"]
        if stratum["status"] == "attached"
    }
    attached_mask = (
        pd.to_numeric(selected_tail["filing_status_code"], errors="raise")
        .astype("int64")
        .isin(attached_codes)
        .to_numpy()
    )
    if not attached_mask.any():
        insufficient = recipient_support["insufficient_support_strata"]
        raise ValueError(
            "PUF capital-gains tail no_attachable_strata: every selected "
            "filing-status stratum has insufficient support; "
            f"insufficient_support={insufficient}."
        )
    tail = selected_tail.loc[attached_mask].reset_index(drop=True)
    assigned_weights = selected_assigned_weights[attached_mask]
    assignments = _assign_tail_donors(
        tail,
        assigned_weights=assigned_weights,
        candidates=candidates,
    )
    # Fidelity is asserted by construction (microcosm#570 review): every
    # joint-vector column in every assignment must equal the SELECTED
    # donor's value, keyed by donor_source_id — reconciliation downstream
    # derives expectations from assignments, so a leak here would
    # self-confirm if it were not caught at the source.
    if sorted(assignments["donor_source_id"].tolist()) != sorted(
        tail["tax_unit_id"].tolist()
    ):
        raise ValueError(
            "PUF tail assignments must consume every selected donor exactly "
            "once (donor-key bijection violated)."
        )
    donor_by_id = tail.set_index("tax_unit_id")
    for column in _JOINT_VECTOR_COLUMNS:
        expected_vector = donor_by_id.loc[
            assignments["donor_source_id"], column
        ].to_numpy(dtype=np.float64)
        assigned_vector = assignments[column].to_numpy(dtype=np.float64)
        if not np.array_equal(expected_vector, assigned_vector):
            raise ValueError(
                "PUF tail assignments leaked recipient values into "
                f"{column}; the selected donor's joint vector must arrive "
                "verbatim."
            )
    before_distribution = _frame_combined_distribution(frame)
    pre_values, pre_weights = _frame_capital_gains_vectors(frame)
    pre_values[_COMBINED_COLUMN] = (
        pre_values["short_term_capital_gains"]
        + pre_values["long_term_capital_gains_before_response"]
    )
    pre_receipts = _raw_top_share_receipts(pre_values, pre_weights)
    transferred, clone_receipt = _clone_and_transfer(
        frame,
        assignments,
    )
    after_distribution = _frame_combined_distribution(transferred)
    post_values, post_weights = _frame_capital_gains_vectors(transferred)
    post_values[_COMBINED_COLUMN] = (
        post_values["short_term_capital_gains"]
        + post_values["long_term_capital_gains_before_response"]
    )
    post_receipts = _raw_top_share_receipts(post_values, post_weights)
    frame_concentration = _frame_capital_gains_concentration_gate(transferred)
    (
        stage_attributable_failures,
        frame_concentration_receipts,
    ) = _stage_attributable_concentration_failures(
        pre_receipts,
        post_receipts,
        frame_concentration,
    )
    if stage_attributable_failures:
        raise ValueError(
            "PUF capital-gains tail transfer worsened weighted concentration "
            "above the threshold:\n  " + "\n  ".join(stage_attributable_failures)
        )
    observed_tail = _observed_tail_transfer(transferred)
    carrier_reconciliation = _reconcile_observed_tail_transfer(
        observed_tail,
        assignments,
    )

    vector_values = {
        column: observed_tail[column].to_numpy(dtype=np.float64)
        for column in _JOINT_VECTOR_COLUMNS
    }
    frame_tail_distribution = _distribution_receipt(
        vector_values["short_term_capital_gains"]
        + vector_values["long_term_capital_gains_before_response"],
        observed_tail["assigned_weight"].to_numpy(dtype=np.float64),
    )
    donor_tail_distribution = _distribution_receipt(
        tail[_TAIL_COMBINED_COLUMN].to_numpy(dtype=np.float64),
        tail["weight"].to_numpy(dtype=np.float64),
    )
    selected_donor_tail_distribution = _distribution_receipt(
        selected_tail[_TAIL_COMBINED_COLUMN].to_numpy(dtype=np.float64),
        selected_tail["weight"].to_numpy(dtype=np.float64),
    )
    signed_reconciliation: dict[str, dict[str, float]] = {}
    for column in (
        "short_term_capital_gains",
        "long_term_capital_gains_before_response",
    ):
        donor_mass = float(
            np.dot(
                tail[column].to_numpy(dtype=np.float64),
                tail["weight"].to_numpy(dtype=np.float64),
            )
        )
        expected = donor_mass * design_weight_normalization
        transferred_mass = float(
            np.dot(
                observed_tail[column].to_numpy(dtype=np.float64),
                observed_tail["assigned_weight"].to_numpy(dtype=np.float64),
            )
        )
        if not np.isclose(transferred_mass, expected, rtol=1e-12, atol=1e-6):
            raise AssertionError(
                f"PUF capital-gains tail {column} signed mass changed during "
                f"joint transfer: expected {expected}, got {transferred_mass}."
            )
        signed_reconciliation[column] = {
            "donor_weighted_signed_mass": donor_mass,
            "design_weight_normalization": float(design_weight_normalization),
            "expected_frame_weighted_signed_mass": expected,
            "transferred_frame_weighted_signed_mass": transferred_mass,
            "difference": float(transferred_mass - expected),
        }

    records = _manifest_records(assignments)
    donor_records = _donor_record_projection(records)
    assignment_records = _assignment_record_projection(records)
    manifest: dict[str, object] = {
        "artifact_kind": "populace_puf_capital_gains_tail_transfer",
        "schema_version": PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION,
        "stage": PUF_CAPITAL_GAINS_TAIL_STAGE_NAME,
        "seed": int(seed),
        "frame_concentration_receipts": frame_concentration_receipts,
        "boundary": {
            **selection,
            "rationale": (
                "Weighted-positive ST+LT q99.5 is below the ASEC recipient "
                "capital-gains topcode while the diagnostic q99.9 reference "
                "is above it; the strict upper stratum therefore carries all "
                "donor mass beyond the recipient support ceiling."
            ),
        },
        "weight_domain": {
            "normalization": (
                "frame_household_design_weight_total / full_donor_design_weight_total"
            ),
            "donor_weight_total": donor_weight_total,
            "frame_household_weight_total": frame_household_weight_total,
            "design_weight_normalization": float(design_weight_normalization),
            "assigned_tail_weight": float(assigned_weights.sum()),
            "selected_tail_weight": float(selected_assigned_weights.sum()),
            "skipped_tail_weight": float(
                selected_assigned_weights.sum() - assigned_weights.sum()
            ),
        },
        "recipient_support": recipient_support,
        "joint_vector_columns": list(_JOINT_VECTOR_COLUMNS),
        "joint_vector_policy": {
            "amount_scale": 1.0,
            "legs_scaled_independently": False,
            "schedule_d_distributions": (
                "derived_by_following_capital_gain_distributions_stage"
            ),
        },
        "clone": clone_receipt,
        "carrier_reconciliation": carrier_reconciliation,
        "tail_distribution_receipts": {
            "donor": donor_tail_distribution,
            "selected_donor": selected_donor_tail_distribution,
            "frame_transferred": frame_tail_distribution,
            "frame_before_stage": before_distribution,
            "frame_after_stage": after_distribution,
        },
        "signed_leg_reconciliation": signed_reconciliation,
        "tail_concentration_gate": {
            "passed": bool(concentration.passed),
            "failures": list(concentration.failures),
            "details": dict(concentration.details),
        },
        "frame_after_stage_concentration_gate": {
            "passed": bool(frame_concentration.passed),
            "failures": list(frame_concentration.failures),
            "details": dict(frame_concentration.details),
        },
        "donor_records_sha256": _canonical_sha256(donor_records),
        "assignment_sha256": _canonical_sha256(assignment_records),
        "record_count": int(len(records)),
        "records": records,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return transferred, manifest


def write_puf_capital_gains_tail_manifest(
    path: Path,
    manifest: Mapping[str, object],
) -> str:
    """Atomically write a verified tail manifest and return its file SHA-256."""

    validate_puf_capital_gains_tail_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def _validate_recipient_support_receipt(
    receipt: object,
    *,
    records: Sequence[Mapping[str, object]] | None,
    selected_donor_count: int | None,
) -> None:
    if not isinstance(receipt, Mapping):
        raise ValueError(
            "PUF capital-gains tail manifest recipient support must be an object."
        )
    support = dict(receipt)
    claimed = support.pop("sha256", None)
    actual = _canonical_sha256(support)
    if claimed != actual:
        raise ValueError(
            "PUF capital-gains tail recipient-support SHA mismatch: "
            f"claimed {claimed!r}, computed {actual!r}."
        )
    expected_fields = {
        "contract",
        "candidate_count",
        "selected_donor_count",
        "attached_donor_count",
        "skipped_donor_count",
        "attached_stratum_count",
        "insufficient_support_stratum_count",
        "not_applicable_stratum_count",
        "insufficient_support_strata",
        "strata",
    }
    if set(support) != expected_fields:
        raise ValueError(
            "PUF capital-gains tail recipient-support receipt schema mismatch."
        )
    if support["contract"] != puf_capital_gains_tail_support_contract_identity():
        raise ValueError(
            "PUF capital-gains tail recipient-support contract identity changed."
        )

    def nonnegative_integer(field: str) -> int:
        value = support.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "PUF capital-gains tail recipient-support "
                f"{field} must be a nonnegative integer."
            )
        return value

    strata = support.get("strata")
    if not isinstance(strata, list) or len(strata) != len(_FILING_STATUS_BY_CODE):
        raise ValueError(
            "PUF capital-gains tail recipient-support strata must enumerate "
            "all five filing statuses exactly once."
        )
    expected_stratum_fields = {
        "filing_status_code",
        "filing_status",
        "status",
        "observed_count",
        "required_minimum",
        "attached_donor_count",
        "skipped_donor_count",
    }
    required_total = 0
    observed_total = 0
    attached_total = 0
    skipped_total = 0
    attached_strata = 0
    insufficient: list[str] = []
    not_applicable = 0
    attached_by_code: dict[int, int] = {}
    for (expected_code, expected_name), raw_stratum in zip(
        _FILING_STATUS_BY_CODE.items(),
        strata,
        strict=True,
    ):
        if not isinstance(raw_stratum, Mapping) or set(raw_stratum) != (
            expected_stratum_fields
        ):
            raise ValueError(
                "PUF capital-gains tail recipient-support stratum schema mismatch."
            )
        stratum = dict(raw_stratum)
        if (
            stratum["filing_status_code"] != expected_code
            or stratum["filing_status"] != expected_name
        ):
            raise ValueError(
                "PUF capital-gains tail recipient-support filing-status order "
                "or identity changed."
            )
        counts: dict[str, int] = {}
        for field in (
            "observed_count",
            "required_minimum",
            "attached_donor_count",
            "skipped_donor_count",
        ):
            value = stratum[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "PUF capital-gains tail recipient-support stratum counts "
                    "must be nonnegative integers."
                )
            counts[field] = value
        observed = counts["observed_count"]
        required = counts["required_minimum"]
        attached = counts["attached_donor_count"]
        skipped = counts["skipped_donor_count"]
        status = stratum["status"]
        if required == 0:
            valid = status == "not_applicable" and attached == skipped == 0
            not_applicable += 1
        elif observed < required:
            valid = (
                status == "insufficient_support"
                and attached == 0
                and skipped == required
            )
            insufficient.append(expected_name)
        else:
            valid = status == "attached" and attached == required and skipped == 0
            attached_strata += 1
        if not valid:
            raise ValueError(
                "PUF capital-gains tail recipient-support status/count "
                f"arithmetic is inconsistent for {expected_name}."
            )
        required_total += required
        observed_total += observed
        attached_total += attached
        skipped_total += skipped
        attached_by_code[expected_code] = attached

    if nonnegative_integer("candidate_count") != observed_total:
        raise ValueError(
            "PUF capital-gains tail recipient-support candidate count does not "
            "equal its strata."
        )
    receipt_selected = nonnegative_integer("selected_donor_count")
    if receipt_selected != required_total or (
        selected_donor_count is not None and receipt_selected != selected_donor_count
    ):
        raise ValueError(
            "PUF capital-gains tail recipient-support selected donor count does "
            "not equal its declared requirement."
        )
    expected_summary = {
        "attached_donor_count": attached_total,
        "skipped_donor_count": skipped_total,
        "attached_stratum_count": attached_strata,
        "insufficient_support_stratum_count": len(insufficient),
        "not_applicable_stratum_count": not_applicable,
    }
    for field, expected in expected_summary.items():
        if nonnegative_integer(field) != expected:
            raise ValueError(
                "PUF capital-gains tail recipient-support summary arithmetic "
                f"is inconsistent for {field}."
            )
    if support.get("insufficient_support_strata") != insufficient:
        raise ValueError(
            "PUF capital-gains tail insufficient-support stratum names changed."
        )
    if records is not None:
        record_counts: Counter[int] = Counter()
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(
                    "PUF capital-gains tail manifest records must contain objects."
                )
            try:
                code = int(record["donor_filing_status_code"])
                name = str(record["donor_filing_status"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "PUF capital-gains tail record filing status is malformed."
                ) from error
            if _FILING_STATUS_BY_CODE.get(code) != name:
                raise ValueError(
                    "PUF capital-gains tail record filing-status identity changed."
                )
            record_counts[code] += 1
        if dict(record_counts) != {
            code: count for code, count in attached_by_code.items() if count
        }:
            raise ValueError(
                "PUF capital-gains tail attached records do not equal the "
                "recipient-support status counts."
            )


def puf_capital_gains_tail_terminal_support_receipt(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Project one validated tail-support receipt into sealed terminal gates."""

    validate_puf_capital_gains_tail_manifest(manifest)
    payload: dict[str, object] = {
        "artifact_kind": "populace_puf_capital_gains_tail_terminal_support",
        "tail_manifest_schema_version": PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION,
        "tail_manifest_sha256": manifest["manifest_sha256"],
        "recipient_support": json.loads(
            json.dumps(manifest["recipient_support"], allow_nan=False)
        ),
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def validate_puf_capital_gains_tail_terminal_support_receipt(
    receipt: Mapping[str, object],
) -> str:
    """Fail closed on a mutated terminal projection of tail support."""

    if not isinstance(receipt, Mapping):
        raise ValueError("PUF capital-gains tail terminal support must be an object.")
    payload = dict(receipt)
    claimed = payload.pop("sha256", None)
    if set(payload) != {
        "artifact_kind",
        "tail_manifest_schema_version",
        "tail_manifest_sha256",
        "recipient_support",
    }:
        raise ValueError("PUF capital-gains tail terminal support schema mismatch.")
    if (
        payload["artifact_kind"] != "populace_puf_capital_gains_tail_terminal_support"
        or payload["tail_manifest_schema_version"]
        != PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("PUF capital-gains tail terminal support identity changed.")
    tail_sha = payload["tail_manifest_sha256"]
    if (
        not isinstance(tail_sha, str)
        or len(tail_sha) != 64
        or any(character not in "0123456789abcdef" for character in tail_sha)
    ):
        raise ValueError(
            "PUF capital-gains tail terminal support manifest SHA is malformed."
        )
    _validate_recipient_support_receipt(
        payload["recipient_support"],
        records=None,
        selected_donor_count=None,
    )
    actual = _canonical_sha256(payload)
    if claimed != actual:
        raise ValueError(
            "PUF capital-gains tail terminal-support SHA mismatch: "
            f"claimed {claimed!r}, computed {actual!r}."
        )
    return actual


def validate_puf_capital_gains_tail_manifest(
    manifest: Mapping[str, object],
) -> str:
    """Validate all manifest hashes and return the canonical payload SHA-256."""

    if not isinstance(manifest, Mapping):
        raise ValueError("PUF capital-gains tail manifest must be an object.")
    payload = dict(manifest)
    claimed = payload.pop("manifest_sha256", None)
    if (
        payload.get("artifact_kind") != "populace_puf_capital_gains_tail_transfer"
        or payload.get("schema_version")
        != PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION
        or payload.get("stage") != PUF_CAPITAL_GAINS_TAIL_STAGE_NAME
    ):
        raise ValueError(
            "PUF capital-gains tail manifest artifact/schema/stage identity changed."
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("PUF capital-gains tail manifest records must be a list.")
    record_count = payload.get("record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count != len(records)
    ):
        raise ValueError("PUF capital-gains tail manifest record count changed.")
    donor_records_sha256 = _canonical_sha256(_donor_record_projection(records))
    if payload.get("donor_records_sha256") != donor_records_sha256:
        raise ValueError(
            "PUF capital-gains tail donor-record SHA mismatch: "
            f"claimed {payload.get('donor_records_sha256')!r}, "
            f"computed {donor_records_sha256!r}."
        )
    assignment_sha256 = _canonical_sha256(_assignment_record_projection(records))
    if payload.get("assignment_sha256") != assignment_sha256:
        raise ValueError(
            "PUF capital-gains tail assignment SHA mismatch: "
            f"claimed {payload.get('assignment_sha256')!r}, "
            f"computed {assignment_sha256!r}."
        )
    boundary = payload.get("boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("PUF capital-gains tail manifest boundary is malformed.")
    selected_donor_count = boundary.get("tail_record_count")
    if (
        isinstance(selected_donor_count, bool)
        or not isinstance(selected_donor_count, int)
        or selected_donor_count <= 0
    ):
        raise ValueError(
            "PUF capital-gains tail manifest selected donor count is malformed."
        )
    _validate_recipient_support_receipt(
        payload.get("recipient_support"),
        records=records,
        selected_donor_count=selected_donor_count,
    )
    actual = _canonical_sha256(payload)
    if claimed != actual:
        raise ValueError(
            "PUF capital-gains tail manifest payload SHA mismatch: "
            f"claimed {claimed!r}, computed {actual!r}."
        )
    return actual


def _recipient_candidates(
    frame: Frame,
    *,
    maximum_transfer_weight: float,
    seed: int,
) -> pd.DataFrame:
    person = frame.table("person")
    household = frame.table("household")
    tax_unit = frame.table("tax_unit")
    person_clone_index = support_clone_index_column("person")
    household_clone_index = support_clone_index_column("household")
    tax_unit_clone_index = support_clone_index_column("tax_unit")
    for table, column in (
        (person, person_clone_index),
        (household, household_clone_index),
        (tax_unit, tax_unit_clone_index),
    ):
        if column not in table:
            raise ValueError(
                f"PUF capital-gains tail transfer requires metadata column {column!r}."
            )

    puf_people = person.loc[puf_tax_detail_clone_mask(person, entity="person")]
    tax_to_household_counts = puf_people.groupby(
        "person_tax_unit_id",
        sort=False,
    )["person_household_id"].nunique()
    if (tax_to_household_counts != 1).any():
        raise ValueError("A PUF tax unit spans multiple recipient households.")
    tax_to_household = puf_people.groupby(
        "person_tax_unit_id",
        sort=False,
    )["person_household_id"].first()

    puf_tax_units = tax_unit.loc[
        puf_tax_detail_clone_mask(tax_unit, entity="tax_unit")
    ].copy()
    puf_tax_units.rename(
        columns={"tax_unit_id": "recipient_tax_unit_id"},
        inplace=True,
    )
    puf_tax_units["recipient_household_id"] = puf_tax_units[
        "recipient_tax_unit_id"
    ].map(tax_to_household)
    if puf_tax_units["recipient_household_id"].isna().any():
        raise ValueError("A PUF recipient tax unit has no household membership.")
    household_tax_unit_count = puf_tax_units.groupby(
        "recipient_household_id",
        sort=False,
    ).size()
    # A one-tax-unit recipient makes the exact donor-grain receipt structural:
    # no secondary copied tax unit can add CG to the transferred slice.
    puf_tax_units = puf_tax_units.loc[
        puf_tax_units["recipient_household_id"].map(household_tax_unit_count) == 1
    ].copy()

    household_ids = household["household_id"].to_numpy(dtype=np.int64)
    household_weights = pd.Series(
        frame.weights_for("household").values,
        index=household_ids,
    )
    puf_household_ids = set(
        household.loc[
            puf_tax_detail_clone_mask(household, entity="household"),
            "household_id",
        ].astype("int64")
    )
    puf_tax_units = puf_tax_units.loc[
        puf_tax_units["recipient_household_id"].isin(puf_household_ids)
    ].copy()
    puf_tax_units["recipient_household_weight"] = puf_tax_units[
        "recipient_household_id"
    ].map(household_weights)
    puf_tax_units = puf_tax_units.loc[
        puf_tax_units["recipient_household_weight"] >= maximum_transfer_weight
    ].copy()

    filing_status = puf_tax_units["filing_status_input"].map(_normalized_filing_status)
    puf_tax_units["recipient_filing_status"] = filing_status
    puf_tax_units["recipient_filing_status_code"] = filing_status.map(
        _FILING_STATUS_CODE
    )
    if puf_tax_units["recipient_filing_status_code"].isna().any():
        bad = sorted(
            set(
                puf_tax_units.loc[
                    puf_tax_units["recipient_filing_status_code"].isna(),
                    "recipient_filing_status",
                ]
            )
        )
        raise ValueError(f"Unknown PUF recipient filing status values: {bad}.")

    proxy_columns = [
        column for column in _RECIPIENT_AGI_PROXY_COLUMNS if column in person
    ]
    if proxy_columns:
        proxy = (
            person.groupby("person_tax_unit_id", sort=False)[proxy_columns]
            .sum()
            .sum(axis=1)
        )
        proxy_values = puf_tax_units["recipient_tax_unit_id"].map(proxy).fillna(0.0)
    else:
        proxy_values = pd.Series(
            np.zeros(len(puf_tax_units), dtype=np.float64),
            index=puf_tax_units.index,
        )
    puf_tax_units["recipient_agi_proxy"] = proxy_values.to_numpy(dtype=np.float64)
    puf_tax_units["recipient_agi_band_index"] = _agi_band_indices(
        puf_tax_units["recipient_agi_proxy"].to_numpy(dtype=np.float64)
    )
    puf_tax_units["recipient_agi_band"] = [
        US_PUF_E19200_AGI_BANDS[index].label
        for index in puf_tax_units["recipient_agi_band_index"]
    ]
    puf_tax_units["recipient_household_source_id"] = puf_tax_units[
        "recipient_household_id"
    ].map(
        pd.Series(
            household[support_source_id_column("household")].to_numpy(),
            index=household_ids,
        )
    )
    puf_tax_units["recipient_tax_unit_source_id"] = puf_tax_units[
        support_source_id_column("tax_unit")
    ].to_numpy()
    puf_tax_units.sort_values(
        ["recipient_household_source_id", "recipient_tax_unit_source_id"],
        kind="mergesort",
        inplace=True,
    )
    rng = np.random.default_rng(seed)
    puf_tax_units["seeded_order"] = rng.random(len(puf_tax_units))
    return puf_tax_units.reset_index(drop=True)


def _recipient_support_receipt(
    tail: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    """Count the declared support universe before any status is attached."""

    donor_codes = pd.to_numeric(tail["filing_status_code"], errors="raise").astype(
        "int64"
    )
    candidate_codes = pd.to_numeric(
        candidates["recipient_filing_status_code"],
        errors="raise",
    ).astype("int64")
    unknown_donor_codes = sorted(set(donor_codes) - set(_FILING_STATUS_BY_CODE))
    unknown_candidate_codes = sorted(set(candidate_codes) - set(_FILING_STATUS_BY_CODE))
    if unknown_donor_codes or unknown_candidate_codes:
        raise ValueError(
            "PUF capital-gains tail support audit found unknown filing-status "
            f"codes: donors={unknown_donor_codes}, "
            f"candidates={unknown_candidate_codes}."
        )
    if candidates["recipient_household_id"].duplicated().any():
        raise ValueError(
            "PUF capital-gains tail support candidates must be unique households."
        )

    required_counts = donor_codes.value_counts().to_dict()
    observed_counts = candidate_codes.value_counts().to_dict()
    strata: list[dict[str, object]] = []
    for code, name in _FILING_STATUS_BY_CODE.items():
        required = int(required_counts.get(code, 0))
        observed = int(observed_counts.get(code, 0))
        if required == 0:
            status = "not_applicable"
            attached = 0
            skipped = 0
        elif observed < required:
            status = "insufficient_support"
            attached = 0
            skipped = required
        else:
            status = "attached"
            attached = required
            skipped = 0
        strata.append(
            {
                "filing_status_code": code,
                "filing_status": name,
                "status": status,
                "observed_count": observed,
                "required_minimum": required,
                "attached_donor_count": attached,
                "skipped_donor_count": skipped,
            }
        )

    insufficient = [
        str(stratum["filing_status"])
        for stratum in strata
        if stratum["status"] == "insufficient_support"
    ]
    payload: dict[str, object] = {
        "contract": puf_capital_gains_tail_support_contract_identity(),
        "candidate_count": int(len(candidates)),
        "selected_donor_count": int(len(tail)),
        "attached_donor_count": int(
            sum(int(stratum["attached_donor_count"]) for stratum in strata)
        ),
        "skipped_donor_count": int(
            sum(int(stratum["skipped_donor_count"]) for stratum in strata)
        ),
        "attached_stratum_count": int(
            sum(stratum["status"] == "attached" for stratum in strata)
        ),
        "insufficient_support_stratum_count": len(insufficient),
        "not_applicable_stratum_count": int(
            sum(stratum["status"] == "not_applicable" for stratum in strata)
        ),
        "insufficient_support_strata": insufficient,
        "strata": strata,
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _assign_tail_donors(
    tail: pd.DataFrame,
    *,
    assigned_weights: np.ndarray,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if len(tail) != len(assigned_weights):
        raise ValueError("Assigned tail weights must align with donor rows.")
    queues: dict[tuple[int, int], list[int]] = {}
    ordered_candidates = candidates.sort_values(
        [
            "recipient_filing_status_code",
            "recipient_agi_band_index",
            "seeded_order",
            "recipient_household_source_id",
            "recipient_tax_unit_source_id",
        ],
        kind="mergesort",
    )
    for index, row in ordered_candidates.iterrows():
        key = (
            int(row["recipient_filing_status_code"]),
            int(row["recipient_agi_band_index"]),
        )
        queues.setdefault(key, []).append(int(index))
    for queue in queues.values():
        queue.reverse()

    rows: list[dict[str, object]] = []
    for donor_position, donor_row in tail.iterrows():
        filing_status_code = int(donor_row["filing_status_code"])
        if filing_status_code not in _FILING_STATUS_BY_CODE:
            raise ValueError(
                f"Unknown PUF donor filing status code {filing_status_code}."
            )
        donor_band = int(donor_row[_TAIL_AGI_BAND_INDEX_COLUMN])
        candidate_index: int | None = None
        for band in sorted(
            range(len(US_PUF_E19200_AGI_BANDS)),
            key=lambda value: (abs(value - donor_band), value),
        ):
            queue = queues.get((filing_status_code, band))
            if queue:
                candidate_index = queue.pop()
                break
        if candidate_index is None:
            raise ValueError(
                "Insufficient unique, single-tax-unit PUF recipient households "
                f"for filing status {_FILING_STATUS_BY_CODE[filing_status_code]!r}."
            )
        candidate = candidates.loc[candidate_index]
        row = donor_row.to_dict()
        row["donor_source_id"] = int(donor_row["tax_unit_id"])
        del row["tax_unit_id"]
        overlap = set(row) & set(candidate.index)
        undeclared = (
            overlap - set(_JOINT_VECTOR_COLUMNS) - _RECIPIENT_OWNED_CANDIDATE_OVERLAP
        )
        if undeclared:
            raise ValueError(
                "PUF tail candidate merge met undeclared donor/candidate "
                f"column overlap: {sorted(undeclared)}. Declare ownership "
                "(donor joint vector vs recipient-owned) before merging — "
                "an undeclared collision silently replaces donor payload "
                "(microcosm#570)."
            )
        row.update(candidate.to_dict())
        # The candidate carries the recipient's EXISTING tax-unit values for
        # joint-vector columns held at tax-unit grain, so the merge above
        # would silently replace the donor's transferred vector with the
        # recipient's old value (microcosm#570 review, Critical: 99.7% of
        # donor unrecaptured-1250 mass was lost this way). The selected
        # donor's joint vector always wins.
        for column in _JOINT_VECTOR_COLUMNS:
            row[column] = donor_row[column]
        row["assigned_weight"] = float(assigned_weights[donor_position])
        rows.append(row)
    result = pd.DataFrame(rows)
    if result["recipient_household_id"].duplicated().any():
        raise AssertionError("PUF tail allocator reused a recipient household.")
    return result


def _clone_and_transfer(
    frame: Frame,
    assignments: pd.DataFrame,
) -> tuple[Frame, dict[str, object]]:
    selected_household_ids = set(assignments["recipient_household_id"].astype("int64"))
    person = frame.table("person")
    selected_person_mask = (
        person["person_household_id"].isin(selected_household_ids).to_numpy()
    )
    selected_people = person.loc[selected_person_mask]
    if selected_people.empty:
        raise ValueError("PUF tail recipient selection contains no people.")

    selected_ids: dict[str, set[int]] = {
        "person": set(selected_people["person_id"].astype("int64"))
    }
    for group in frame.schema.group_entities:
        selected_ids[group] = set(
            selected_people[frame.schema.membership_column(group)].astype("int64")
        )
        membership = frame.schema.membership_column(group)
        selected_group_mask = person[membership].isin(selected_ids[group])
        selected_group_members = pd.DataFrame(
            {
                "group_id": person.loc[selected_group_mask, membership].to_numpy(),
                "household_id": person.loc[
                    selected_group_mask,
                    "person_household_id",
                ].to_numpy(),
            }
        )
        household_counts = selected_group_members.groupby(
            "group_id",
            sort=False,
        )["household_id"].nunique()
        crosses_household_boundary = (
            len(selected_group_members) != int(selected_person_mask.sum())
            or (household_counts != 1).any()
        )
        if crosses_household_boundary:
            raise ValueError(
                f"PUF tail recipient {group} membership crosses a selected "
                "household boundary; refusing a partial graph clone."
            )

    id_multiplier = _support_clone_multiplier(frame)
    tables: dict[str, pd.DataFrame] = {}
    selected_masks: dict[str, np.ndarray] = {}
    before_rows: dict[str, int] = {}
    after_rows: dict[str, int] = {}
    for entity in frame.entities:
        table = frame.table(entity).copy()
        primary = frame.schema.entity_id_column(entity)
        selected_mask = table[primary].isin(selected_ids[entity]).to_numpy()
        selected_masks[entity] = selected_mask
        clone = table.loc[selected_mask].copy()
        if clone.empty:
            raise ValueError(f"PUF tail selection has no {entity} rows to clone.")
        clone_index = support_clone_index_column(entity)
        if not (
            clone[clone_index].to_numpy(dtype=np.int64) == PUF_TAX_DETAIL_CLONE_INDEX
        ).all():
            raise ValueError(f"PUF tail selected a non-primary PUF {entity} clone.")
        clone[primary] = clone[primary].to_numpy(dtype=np.int64) + id_multiplier
        if entity == frame.schema.person_entity:
            for group in frame.schema.group_entities:
                membership = frame.schema.membership_column(group)
                clone[membership] = (
                    clone[membership].to_numpy(dtype=np.int64) + id_multiplier
                )
        clone[clone_index] = 2
        combined = pd.concat([table, clone], ignore_index=True)
        if combined[primary].duplicated().any():
            raise ValueError(f"PUF tail cloned duplicate {primary} values.")
        if (
            entity != frame.schema.person_entity
            and not combined[primary].is_monotonic_increasing
        ):
            raise ValueError(f"PUF tail clone broke sorted {primary} order.")
        tables[entity] = combined
        before_rows[entity] = int(len(table))
        after_rows[entity] = int(len(combined))

    tax_unit = tables["tax_unit"]
    provenance_defaults: Mapping[str, Any] = {
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN: False,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN: -1,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN: False,
        PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN: 0,
        PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN: -1,
        PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN: 0.0,
    }
    original_tax_unit_rows = before_rows["tax_unit"]
    for column, default in provenance_defaults.items():
        if column in tax_unit:
            raise ValueError(
                f"PUF capital-gains tail provenance column {column!r} already exists."
            )
        tax_unit[column] = default

    tail_person_mask = (
        tables["person"][support_clone_index_column("person")].to_numpy(dtype=np.int64)
        == 2
    )
    for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
        tables["person"].loc[tail_person_mask, column] = 0.0
    tail_tax_unit_mask = np.arange(len(tax_unit)) >= original_tax_unit_rows
    for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
        tax_unit.loc[tail_tax_unit_mask, column] = 0.0

    tail_person_positions = np.flatnonzero(tail_person_mask)
    tail_person_tax_unit_ids = (
        tables["person"]
        .loc[
            tail_person_mask,
            "person_tax_unit_id",
        ]
        .to_numpy(dtype=np.int64)
    )
    first_person_position_by_tax_unit = (
        pd.Series(tail_person_positions, index=tail_person_tax_unit_ids)
        .groupby(level=0, sort=False)
        .first()
        .to_dict()
    )
    tax_unit_position_by_id = pd.Series(
        np.arange(len(tax_unit), dtype=np.int64),
        index=tax_unit["tax_unit_id"].to_numpy(dtype=np.int64),
    ).to_dict()

    clone_details: dict[int, dict[str, object]] = {}
    for assignment_index, assignment in assignments.iterrows():
        recipient_tax_unit_id = int(assignment["recipient_tax_unit_id"])
        recipient_household_id = int(assignment["recipient_household_id"])
        tail_tax_unit_id = recipient_tax_unit_id + id_multiplier
        tail_household_id = recipient_household_id + id_multiplier
        first_person_position = first_person_position_by_tax_unit.get(tail_tax_unit_id)
        if first_person_position is None:
            raise AssertionError("PUF tail clone has no target-tax-unit person.")
        first_person_position = int(first_person_position)
        for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
            tables["person"].at[first_person_position, column] = float(
                assignment[column]
            )
        target_tax_unit_position = tax_unit_position_by_id.get(tail_tax_unit_id)
        if target_tax_unit_position is None:
            raise AssertionError("PUF tail clone target tax unit is not present.")
        target_tax_unit_position = int(target_tax_unit_position)
        for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
            tax_unit.at[target_tax_unit_position, column] = float(assignment[column])
        tax_unit.at[
            target_tax_unit_position,
            PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        ] = True
        tax_unit.at[
            target_tax_unit_position,
            PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
        ] = int(assignment["donor_source_id"])
        tax_unit.at[
            target_tax_unit_position,
            PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
        ] = bool(assignment[_TAIL_SYNTHETIC_COLUMN])
        tax_unit.at[
            target_tax_unit_position,
            PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
        ] = int(assignment["filing_status_code"])
        tax_unit.at[
            target_tax_unit_position,
            PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
        ] = int(assignment[_TAIL_AGI_BAND_INDEX_COLUMN])
        tax_unit.at[
            target_tax_unit_position,
            PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
        ] = float(assignment["assigned_weight"])
        clone_details[int(assignment_index)] = {
            "recipient_household_id": recipient_household_id,
            "recipient_tax_unit_id": recipient_tax_unit_id,
            "tail_household_id": tail_household_id,
            "tail_tax_unit_id": tail_tax_unit_id,
            "tail_person_id": int(
                tables["person"].at[first_person_position, "person_id"]
            ),
        }

    household = frame.table("household")
    household_ids = household["household_id"].to_numpy(dtype=np.int64)
    assignment_by_household = assignments.set_index("recipient_household_id")[
        "assigned_weight"
    ]
    old_weights = frame.weights_for("household")
    remaining_weights = old_weights.values.copy()
    household_position_by_id = pd.Series(
        np.arange(len(household), dtype=np.int64),
        index=household_ids,
    ).to_dict()
    for household_id, assigned_weight in assignment_by_household.items():
        position = household_position_by_id.get(int(household_id))
        if position is None:
            raise AssertionError("PUF tail recipient household is not present.")
        position = int(position)
        remaining_weights[position] -= float(assigned_weight)
        if remaining_weights[position] < -1e-10:
            raise ValueError(
                f"PUF tail transfer exceeds household {household_id} weight."
            )
        remaining_weights[position] = max(remaining_weights[position], 0.0)
    cloned_household_ids = household.loc[
        selected_masks["household"],
        "household_id",
    ].astype("int64")
    clone_weights = cloned_household_ids.map(assignment_by_household).to_numpy(
        dtype=np.float64
    )
    new_weights = Weights(
        np.concatenate([remaining_weights, clone_weights]),
        old_weights.kind,
    )
    old_weights.assert_mass_conserved(new_weights)

    transferred = Frame(
        tables,
        frame.schema,
        {"household": new_weights},
        pd.concat(
            [
                frame.strata,
                frame.strata.loc[selected_person_mask].reset_index(drop=True),
            ],
            ignore_index=True,
        ),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    effective_group_weight_before: dict[str, float] = {}
    effective_group_weight_after: dict[str, float] = {}
    effective_group_weight_difference: dict[str, float] = {}
    for group in frame.schema.group_entities:
        before = frame.resolve_weights(group).total
        after = transferred.resolve_weights(group).total
        if not np.isclose(before, after, rtol=1e-12, atol=1e-7):
            raise AssertionError(
                "PUF capital-gains tail changed effective "
                f"{group} weight mass: {before} -> {after}."
            )
        effective_group_weight_before[group] = before
        effective_group_weight_after[group] = after
        effective_group_weight_difference[group] = float(after - before)

    for assignment_index, details in clone_details.items():
        for key, value in details.items():
            assignments.loc[assignment_index, key] = value
    source_before = assignments["recipient_household_id"].map(
        pd.Series(old_weights.values, index=household_ids)
    )
    assignments["recipient_household_weight_before"] = source_before
    assignments["recipient_household_weight_after"] = (
        source_before - assignments["assigned_weight"]
    )
    clone_receipt: dict[str, object] = {
        "method": "selective_post_qrf_household_weight_split",
        "support_channel": PUF_CAPITAL_GAINS_TAIL_SUPPORT_CHANNEL,
        "clone_index": 2,
        "id_multiplier": int(id_multiplier),
        "selected_household_count": int(len(assignments)),
        "household_weight_total_before": old_weights.total,
        "household_weight_total_after": new_weights.total,
        "household_weight_difference": float(new_weights.total - old_weights.total),
        "effective_tax_unit_weight_total_before": effective_group_weight_before[
            "tax_unit"
        ],
        "effective_tax_unit_weight_total_after": effective_group_weight_after[
            "tax_unit"
        ],
        "effective_group_weight_totals_before": effective_group_weight_before,
        "effective_group_weight_totals_after": effective_group_weight_after,
        "effective_group_weight_differences": effective_group_weight_difference,
        "entity_rows_before": before_rows,
        "entity_rows_after": after_rows,
    }
    return transferred, clone_receipt


def _support_clone_multiplier(frame: Frame) -> int:
    """Recover one clone offset from source-matched native/detail pairs.

    Seeded attachment keeps every native row but only a selected subset of the
    primary PUF clone. Counts and positional ordering therefore need not match;
    the immutable support source ID is the pairing authority.
    """

    multiplier: int | None = None
    for entity in frame.entities:
        table = frame.table(entity)
        clone_index = support_clone_index_column(entity)
        source_id = support_source_id_column(entity)
        missing = [column for column in (clone_index, source_id) if column not in table]
        if missing:
            raise ValueError(f"PUF support {entity} metadata missing: {missing}.")
        clone_indices = table[clone_index].to_numpy(dtype=np.int64)
        if (clone_indices >= 2).any():
            raise ValueError("PUF capital-gains tail transfer must run exactly once.")
        puf = puf_tax_detail_clone_mask(table, entity=entity)
        if not puf.any():
            raise ValueError(f"PUF detail clone has no {entity} rows.")
        native = clone_indices == 0
        primary = frame.schema.entity_id_column(entity)
        source_ids = table[source_id].to_numpy(dtype=np.int64)
        native_source_ids = source_ids[native]
        puf_source_ids = source_ids[puf]
        if len(np.unique(native_source_ids)) != len(native_source_ids):
            raise ValueError(f"PUF support {entity} native source IDs are not unique.")
        if len(np.unique(puf_source_ids)) != len(puf_source_ids):
            raise ValueError(
                f"PUF support {entity} detail clone source IDs are not unique."
            )
        native_primary_by_source = dict(
            zip(
                native_source_ids.tolist(),
                table.loc[native, primary].to_numpy(dtype=np.int64).tolist(),
                strict=True,
            )
        )
        missing_native_sources = sorted(
            set(puf_source_ids.tolist()) - set(native_primary_by_source)
        )
        if missing_native_sources:
            raise ValueError(
                f"PUF support {entity} detail clone source IDs have no native "
                f"match: {missing_native_sources}."
            )
        matched_native_ids = np.asarray(
            [native_primary_by_source[source] for source in puf_source_ids],
            dtype=np.int64,
        )
        puf_ids = table.loc[puf, primary].to_numpy(dtype=np.int64)
        differences = puf_ids - matched_native_ids
        unique = np.unique(differences)
        if len(unique) != 1 or int(unique[0]) <= 0:
            raise ValueError(f"PUF support {entity} IDs do not share one clone offset.")
        candidate = int(unique[0])
        if multiplier is None:
            multiplier = candidate
        elif multiplier != candidate:
            raise ValueError("PUF support entities use inconsistent clone offsets.")
    assert multiplier is not None
    return multiplier


def _frame_combined_distribution(frame: Frame) -> dict[str, object]:
    values, weights = _frame_capital_gains_vectors(frame)
    combined = (
        values["short_term_capital_gains"]
        + values["long_term_capital_gains_before_response"]
    )
    receipt = _distribution_receipt(
        combined,
        weights,
    )
    positive_mass = float(receipt["weighted_positive_mass"])
    five_x_ceiling = 5.0 * positive_mass
    target = PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET
    receipt["positive_mass_five_x_ceiling"] = five_x_ceiling
    receipt["positive_mass_five_x_target"] = target
    receipt["positive_mass_five_x_headroom"] = five_x_ceiling - target
    receipt["positive_mass_five_x_target_ratio"] = five_x_ceiling / target
    receipt["positive_mass_five_x_target_exceeded"] = five_x_ceiling > target
    return receipt


def _frame_capital_gains_concentration_gate(frame: Frame) -> GateResult:
    values, weights = _frame_capital_gains_vectors(frame)
    values[_COMBINED_COLUMN] = (
        values["short_term_capital_gains"]
        + values["long_term_capital_gains_before_response"]
    )
    return tail_concentration_gate(
        values,
        {column: weights for column in values},
        top_k=PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_TOP_K,
        max_top_share=PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MAX_TOP_SHARE,
        min_nonzero_records=(PUF_CAPITAL_GAINS_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS),
        reviewed_exclusions={},
    )


def _frame_capital_gains_vectors(
    frame: Frame,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    person = frame.table("person")
    missing_person = sorted(
        {"person_tax_unit_id", *PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS}
        - set(person.columns)
    )
    if missing_person:
        raise ValueError(
            f"Frame capital-gains receipt missing person columns: {missing_person}."
        )
    tax_unit = frame.table("tax_unit")
    missing_tax_unit = sorted(
        {"tax_unit_id", *PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS}
        - set(tax_unit.columns)
    )
    if missing_tax_unit:
        raise ValueError(
            f"Frame capital-gains receipt missing tax-unit columns: {missing_tax_unit}."
        )
    tax_unit_ids = tax_unit["tax_unit_id"].to_numpy(dtype=np.int64)
    person_vectors = person.groupby("person_tax_unit_id", sort=False)[
        list(PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS)
    ].sum()
    person_vectors = person_vectors.reindex(tax_unit_ids)
    if person_vectors.isna().any().any():
        raise ValueError("Frame capital-gains receipt found an empty tax unit.")
    values = {
        column: person_vectors[column].to_numpy(dtype=np.float64)
        for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS
    }
    values.update(
        {
            column: tax_unit[column].to_numpy(dtype=np.float64)
            for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS
        }
    )
    return values, frame.resolve_weights("tax_unit").values


def _observed_tail_transfer(frame: Frame) -> pd.DataFrame:
    """Read the materialized tail vector and effective weights from the frame."""

    tax_unit = frame.table("tax_unit")
    required_tax_unit = {
        "tax_unit_id",
        "filing_status_input",
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
        PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
        *PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
    }
    missing_tax_unit = sorted(required_tax_unit - set(tax_unit.columns))
    if missing_tax_unit:
        raise ValueError(
            "Materialized PUF capital-gains tail tax-unit columns missing: "
            f"{missing_tax_unit}."
        )
    applied = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].astype(bool).to_numpy()
    if not applied.any():
        raise ValueError("Materialized PUF capital-gains tail has no applied rows.")
    observed = tax_unit.loc[
        applied,
        [
            "tax_unit_id",
            "filing_status_input",
            PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
            PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
            PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
            PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
            PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
            *PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
        ],
    ].copy()
    observed.rename(
        columns={
            "tax_unit_id": "tail_tax_unit_id",
            PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN: "donor_source_id",
            PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN: "donor_is_synthetic",
            PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN: (
                "donor_filing_status_code"
            ),
            PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN: "donor_agi_band_index",
            PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN: (
                "transfer_weight_provenance"
            ),
        },
        inplace=True,
    )
    observed["filing_status_input"] = observed["filing_status_input"].map(
        _normalized_filing_status
    )
    observed["assigned_weight"] = frame.resolve_weights("tax_unit").values[applied]

    person = frame.table("person")
    missing_person = sorted(
        {"person_tax_unit_id", *PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS}
        - set(person.columns)
    )
    if missing_person:
        raise ValueError(
            "Materialized PUF capital-gains tail person columns missing: "
            f"{missing_person}."
        )
    person_vectors = person.groupby("person_tax_unit_id", sort=False)[
        list(PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS)
    ].sum()
    tail_tax_unit_ids = observed["tail_tax_unit_id"].to_numpy(dtype=np.int64)
    person_vectors = person_vectors.reindex(tail_tax_unit_ids)
    if person_vectors.isna().any().any():
        raise ValueError("A materialized PUF capital-gains tail tax unit is empty.")
    for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
        observed[column] = person_vectors[column].to_numpy(dtype=np.float64)
    observed.sort_values("donor_source_id", kind="mergesort", inplace=True)
    observed.reset_index(drop=True, inplace=True)
    return observed


def _reconcile_observed_tail_transfer(
    observed: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, object]:
    """Assert that exact joint assignments reached the materialized carrier."""

    expected = assignments[
        [
            "donor_source_id",
            "tail_tax_unit_id",
            "assigned_weight",
            _TAIL_SYNTHETIC_COLUMN,
            "filing_status_code",
            _TAIL_AGI_BAND_INDEX_COLUMN,
            *_JOINT_VECTOR_COLUMNS,
        ]
    ].copy()
    expected.rename(
        columns={
            _TAIL_SYNTHETIC_COLUMN: "donor_is_synthetic",
            "filing_status_code": "donor_filing_status_code",
            _TAIL_AGI_BAND_INDEX_COLUMN: "donor_agi_band_index",
        },
        inplace=True,
    )
    expected["filing_status_input"] = expected["donor_filing_status_code"].map(
        _FILING_STATUS_BY_CODE
    )
    expected["transfer_weight_provenance"] = assignments["assigned_weight"].to_numpy()
    comparison_columns = [
        "donor_source_id",
        "tail_tax_unit_id",
        "assigned_weight",
        "transfer_weight_provenance",
        "donor_is_synthetic",
        "donor_filing_status_code",
        "donor_agi_band_index",
        "filing_status_input",
        *_JOINT_VECTOR_COLUMNS,
    ]
    expected = expected[comparison_columns].sort_values(
        "donor_source_id",
        kind="mergesort",
    )
    expected.reset_index(drop=True, inplace=True)
    actual = observed[comparison_columns]
    if len(actual) != len(expected):
        raise AssertionError(
            "PUF capital-gains tail materialized carrier count differs from "
            f"assignments: {len(actual)} != {len(expected)}."
        )
    exact_columns = (
        "donor_source_id",
        "tail_tax_unit_id",
        "donor_is_synthetic",
        "donor_filing_status_code",
        "donor_agi_band_index",
        "filing_status_input",
    )
    for column in exact_columns:
        if not np.array_equal(
            actual[column].to_numpy(),
            expected[column].to_numpy(),
        ):
            raise AssertionError(
                "PUF capital-gains tail materialized carrier changed lineage "
                f"field {column}."
            )
    maximum_absolute_differences: dict[str, float] = {}
    for column in (
        "assigned_weight",
        "transfer_weight_provenance",
        *_JOINT_VECTOR_COLUMNS,
    ):
        expected_values = expected[column].to_numpy(dtype=np.float64)
        actual_values = actual[column].to_numpy(dtype=np.float64)
        differences = np.abs(actual_values - expected_values)
        maximum = float(differences.max()) if len(differences) else 0.0
        if maximum != 0.0:
            raise AssertionError(
                "PUF capital-gains tail materialized carrier changed "
                f"{column}: maximum absolute difference {maximum}."
            )
        maximum_absolute_differences[column] = maximum
    return {
        "passed": True,
        "observed_tail_tax_unit_count": int(len(actual)),
        "donor_source_ids_match": True,
        "lineage_fields_match": True,
        "maximum_absolute_differences": maximum_absolute_differences,
    }


def _distribution_receipt(
    values: Sequence[float],
    weights: Sequence[float],
) -> dict[str, object]:
    numeric_values = np.asarray(values, dtype=np.float64)
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if (
        numeric_values.ndim != 1
        or numeric_weights.ndim != 1
        or len(numeric_values) != len(numeric_weights)
    ):
        raise ValueError("Capital-gains distribution values and weights must align.")
    if not np.isfinite(numeric_values).all() or not np.isfinite(numeric_weights).all():
        raise ValueError(
            "Capital-gains distribution values and weights must be finite."
        )
    if (numeric_weights < 0.0).any() or numeric_weights.sum() <= 0.0:
        raise ValueError("Capital-gains distribution weights must be nonnegative.")
    positive = (numeric_values > 0.0) & (numeric_weights > 0.0)
    positive_weight = float(numeric_weights[positive].sum())
    positive_mass = float(np.dot(numeric_values[positive], numeric_weights[positive]))
    absolute_mass = np.abs(numeric_values) * numeric_weights
    nonzero_mass = absolute_mass[absolute_mass > 0.0]
    top_count = min(100, len(nonzero_mass))
    top_mass = (
        float(np.partition(nonzero_mass, -top_count)[-top_count:].sum())
        if top_count
        else 0.0
    )
    quantiles = {
        name: (
            _weighted_quantile(
                numeric_values[positive],
                numeric_weights[positive],
                quantile,
            )
            if positive.any()
            else 0.0
        )
        for name, quantile in (
            ("p90", 0.9),
            ("p99", 0.99),
            ("p99_9", 0.999),
        )
    }
    total_weight = float(numeric_weights.sum())
    return {
        "record_count": int(len(numeric_values)),
        "positive_carrier_count": int(positive.sum()),
        "total_weight": total_weight,
        "positive_carrier_weight": positive_weight,
        "positive_carrier_rate": float(positive_weight / total_weight),
        "weighted_signed_mass": float(np.dot(numeric_values, numeric_weights)),
        "weighted_positive_mass": positive_mass,
        "conditional_positive_mean": (
            float(positive_mass / positive_weight) if positive_weight else 0.0
        ),
        **quantiles,
        "maximum": (float(numeric_values[positive].max()) if positive.any() else 0.0),
        "top_100_weighted_absolute_mass_share": (
            float(top_mass / nonzero_mass.sum()) if len(nonzero_mass) else 0.0
        ),
    }


def _manifest_records(assignments: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in assignments.sort_values(
        "donor_source_id",
        kind="mergesort",
    ).iterrows():
        assigned_weight = float(row["assigned_weight"])
        joint_vector = {column: float(row[column]) for column in _JOINT_VECTOR_COLUMNS}
        combined = (
            joint_vector["short_term_capital_gains"]
            + joint_vector["long_term_capital_gains_before_response"]
        )
        records.append(
            {
                "donor_source_id": int(row["donor_source_id"]),
                "donor_weight": float(row["weight"]),
                "assigned_weight": assigned_weight,
                "donor_filing_status_code": int(row["filing_status_code"]),
                "donor_filing_status": _FILING_STATUS_BY_CODE[
                    int(row["filing_status_code"])
                ],
                "recipient_filing_status": str(row["recipient_filing_status"]),
                "donor_agi_band_index": int(row[_TAIL_AGI_BAND_INDEX_COLUMN]),
                "donor_agi_band": str(row[_TAIL_AGI_BAND_LABEL_COLUMN]),
                "recipient_agi_band_index": int(row["recipient_agi_band_index"]),
                "recipient_agi_band": str(row["recipient_agi_band"]),
                "agi_band_exact_match": bool(
                    int(row[_TAIL_AGI_BAND_INDEX_COLUMN])
                    == int(row["recipient_agi_band_index"])
                ),
                "donor_is_synthetic": bool(row[_TAIL_SYNTHETIC_COLUMN]),
                "recipient_household_source_id": int(
                    row["recipient_household_source_id"]
                ),
                "recipient_tax_unit_source_id": int(
                    row["recipient_tax_unit_source_id"]
                ),
                "recipient_household_id": int(row["recipient_household_id"]),
                "recipient_tax_unit_id": int(row["recipient_tax_unit_id"]),
                "tail_household_id": int(row["tail_household_id"]),
                "tail_tax_unit_id": int(row["tail_tax_unit_id"]),
                "tail_person_id": int(row["tail_person_id"]),
                "recipient_household_weight_before": float(
                    row["recipient_household_weight_before"]
                ),
                "recipient_household_weight_after": float(
                    row["recipient_household_weight_after"]
                ),
                "joint_vector": joint_vector,
                "weighted_masses": {
                    **{
                        column: float(value * assigned_weight)
                        for column, value in joint_vector.items()
                    },
                    "short_term_plus_long_term_signed": float(
                        combined * assigned_weight
                    ),
                    "short_term_plus_long_term_positive": float(
                        max(combined, 0.0) * assigned_weight
                    ),
                },
            }
        )
    return records


def _donor_record_projection(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    keys = (
        "donor_source_id",
        "donor_weight",
        "donor_filing_status_code",
        "donor_filing_status",
        "donor_agi_band_index",
        "donor_agi_band",
        "donor_is_synthetic",
        "joint_vector",
    )
    return [{key: record[key] for key in keys} for record in records]


def _assignment_record_projection(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    keys = (
        "donor_source_id",
        "assigned_weight",
        "recipient_household_source_id",
        "recipient_tax_unit_source_id",
        "recipient_household_id",
        "recipient_tax_unit_id",
        "tail_household_id",
        "tail_tax_unit_id",
        "tail_person_id",
    )
    return [{key: record[key] for key in keys} for record in records]


def _agi_band_indices(values: Sequence[float]) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise ValueError("PUF tail AGI values must be one-dimensional and finite.")
    return np.searchsorted(_AGI_UPPER_BOUNDS, numeric, side="right")


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    numeric_values = np.asarray(values, dtype=np.float64)
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if len(numeric_values) != len(numeric_weights) or len(numeric_values) == 0:
        raise ValueError(
            "Weighted quantile values and weights must align and be nonempty."
        )
    positive_weight = numeric_weights > 0.0
    numeric_values = numeric_values[positive_weight]
    numeric_weights = numeric_weights[positive_weight]
    order = np.argsort(numeric_values, kind="mergesort")
    ordered_values = numeric_values[order]
    cumulative = np.cumsum(numeric_weights[order])
    position = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def _normalized_filing_status(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
