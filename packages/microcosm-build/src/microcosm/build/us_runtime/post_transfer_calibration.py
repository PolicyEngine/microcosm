"""Deterministic two-part calibration for mutable US transfer draws.

This module is deliberately provenance-blind.  Its callers own the source
spine and pass exact reference, recipient, and mutable row masks.  The kernel
then calibrates only the positive leg: carrier prevalence is either matched to
the weighted reference margin or frozen, and mutable positive amounts are
mapped onto reference support at the terminal battery's five quantiles.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from microcosm.frame import Frame

__all__ = [
    "POST_TRANSFER_CALIBRATION_SPECS",
    "PostTransferCalibrationFrameResult",
    "PostTransferCalibrationResult",
    "PostTransferCalibrationSpec",
    "apply_post_transfer_calibration",
    "calibrate_post_transfer_values",
    "post_transfer_calibration_policy_identity",
    "post_transfer_calibration_spec",
    "post_transfer_calibration_spec_for_target",
    "validate_post_transfer_calibration_receipt",
]

_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_POLICY_ARTIFACT_KIND = "microcosm_us_post_transfer_calibration_policy"
_POLICY_SCHEMA_VERSION = 1
_RECEIPT_SCHEMA_VERSION = 2


def _receipt_verification_contract() -> dict[str, object]:
    """Describe which receipt claims can be replayed from a terminal frame."""

    return {
        "terminal_pre_state_replay": False,
        "terminal_pre_state_reason": (
            "pre-calibration frame is not present at terminal validation"
        ),
        "terminal_live_receipt_paths": [
            "scope.rows",
            "scope.reference_rows",
            "scope.recipient_rows",
            "scope.reference_rows_sha256",
            "scope.recipient_rows_sha256",
            "scope.entity_ids_sha256",
            "scope.output_values_sha256",
            "weights.sha256",
            "weights.reference_total",
            "weights.recipient_total",
            "carrier.reference_positive_mass",
            "carrier.reference_positive_share",
            "carrier.target_positive_mass",
            "carrier.after_positive_mass",
            "carrier.after_positive_share",
            "carrier.residual_after_minus_target",
            "carrier.absolute_residual",
            "amount.reference_quantiles",
            "amount.recipient_after_quantiles",
            "amount.qed_after",
        ],
        "generation_transition_receipt_paths": [
            "scope.mutable_rows",
            "scope.effective_mutable_rows",
            "scope.mutable_rows_sha256",
            "scope.allowed_carrier_rows",
            "scope.allowed_carrier_rows_sha256",
            "scope.allowed_carrier_rows_mode",
            "scope.addition_candidate_rows",
            "scope.addition_candidate_rows_sha256",
            "scope.addition_candidate_rows_mode",
            "scope.input_values_sha256",
            "carrier.before_positive_mass",
            "carrier.before_positive_share",
            "carrier.removed_rows",
            "carrier.added_rows",
            "carrier.disallowed_cleared_rows",
            "carrier.capacity_limited",
            "carrier.capacity",
            "carrier.selection",
            "amount.recipient_before_quantiles",
            "amount.qed_before",
            "amount.mapped_rows",
            "amount.anchor_rows",
            "amount.anchor_conflicts",
            "amount.donor_support_violations",
            "invariants",
        ],
        "frame_owner_only_receipt_paths": [
            "scope.applied_changed_rows",
            "weights.kind",
        ],
        "policy_structural_receipt_paths": [
            "artifact_kind",
            "schema_version",
            "verification_contract",
            "policy_sha256",
            "spec",
            "carrier.mode",
            "amount.quantiles",
            "amount.exact_anchor_count",
            "amount.unanchored_quantiles",
            "amount.status",
        ],
        "generation_transition_binding": (
            "generation-time input/output context and enclosing execution authority"
        ),
    }


@dataclass(frozen=True)
class PostTransferCalibrationSpec:
    """One authority-bound positive-leg post-transfer calibration."""

    entity: str
    family: str
    target: str
    stage: Literal["early_gap_fill", "late_transfer"]
    carrier_mode: Literal["match_reference", "preserve_recipient"]
    negative_leg: Literal["byte_exact"] = "byte_exact"
    special_constraint: Literal[
        "none",
        "adult_care_qualifying_one_per_tax_unit",
        "weeks_requires_positive_unemployment_compensation",
    ] = "none"

    def __post_init__(self) -> None:
        for label, value in (
            ("entity", self.entity),
            ("family", self.family),
            ("target", self.target),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Post-transfer calibration {label} must be non-empty."
                )
        if self.stage not in {"early_gap_fill", "late_transfer"}:
            raise ValueError(f"Unknown post-transfer calibration stage {self.stage!r}.")
        if self.carrier_mode not in {"match_reference", "preserve_recipient"}:
            raise ValueError(
                f"Unknown post-transfer carrier mode {self.carrier_mode!r}."
            )
        if self.negative_leg != "byte_exact":
            raise ValueError("Post-transfer calibration must preserve negatives.")
        if self.special_constraint not in {
            "none",
            "adult_care_qualifying_one_per_tax_unit",
            "weeks_requires_positive_unemployment_compensation",
        }:
            raise ValueError(
                "Unknown post-transfer calibration special constraint "
                f"{self.special_constraint!r}."
            )

    @property
    def key(self) -> str:
        return f"{self.entity}/{self.family}/{self.target}"


def _spec(
    entity: str,
    family: str,
    target: str,
    stage: Literal["early_gap_fill", "late_transfer"],
    carrier_mode: Literal["match_reference", "preserve_recipient"],
    special_constraint: Literal[
        "none",
        "adult_care_qualifying_one_per_tax_unit",
        "weeks_requires_positive_unemployment_compensation",
    ] = "none",
) -> PostTransferCalibrationSpec:
    return PostTransferCalibrationSpec(
        entity=entity,
        family=family,
        target=target,
        stage=stage,
        carrier_mode=carrier_mode,
        special_constraint=special_constraint,
    )


_ORDERED_SPECS = (
    _spec(
        "person",
        "model_required_numeric",
        "unemployment_compensation",
        "early_gap_fill",
        "preserve_recipient",
    ),
    _spec(
        "person",
        "source_operator_prior_year_income",
        "self_employment_income_last_year",
        "early_gap_fill",
        "match_reference",
    ),
    _spec(
        "person",
        "adult_care",
        "pre_subsidy_care_expenses",
        "late_transfer",
        "match_reference",
        "adult_care_qualifying_one_per_tax_unit",
    ),
    _spec(
        "person",
        "source_operator_child_support",
        "child_support_expense",
        "late_transfer",
        "match_reference",
    ),
    _spec(
        "person",
        "source_operator_child_support",
        "child_support_received",
        "late_transfer",
        "match_reference",
    ),
    _spec(
        "person",
        "source_operator_disability_benefits",
        "disability_benefits",
        "late_transfer",
        "preserve_recipient",
    ),
    _spec(
        "person",
        "source_operator_weeks_unemployed",
        "weeks_unemployed",
        "late_transfer",
        "match_reference",
        "weeks_requires_positive_unemployment_compensation",
    ),
    _spec(
        "person",
        "source_operator_workers_compensation",
        "workers_compensation",
        "late_transfer",
        "match_reference",
    ),
    _spec(
        "spm_unit",
        "source_operator_energy_subsidy",
        "spm_unit_energy_subsidy",
        "late_transfer",
        "match_reference",
    ),
)

POST_TRANSFER_CALIBRATION_SPECS = MappingProxyType(
    {spec.key: spec for spec in _ORDERED_SPECS}
)


@dataclass(frozen=True)
class PostTransferCalibrationResult:
    """Calibrated values plus a deterministic audit receipt."""

    values: np.ndarray
    receipt: dict[str, object]


@dataclass(frozen=True)
class PostTransferCalibrationFrameResult:
    """A frame with one calibrated target plus its audit receipt."""

    frame: Frame
    receipt: dict[str, object]


@dataclass(frozen=True)
class _PrefixSchedule:
    """One bound carrier order and its sole float64 accumulation path."""

    ordered_positions: np.ndarray
    cumulative_mass: np.ndarray


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def post_transfer_calibration_policy_identity() -> dict[str, object]:
    """Return the complete JSON policy identity, including its content hash."""

    live_specs: list[PostTransferCalibrationSpec] = []
    if not isinstance(POST_TRANSFER_CALIBRATION_SPECS, Mapping):
        raise ValueError("Post-transfer calibration registry must be a mapping.")
    for key, spec in POST_TRANSFER_CALIBRATION_SPECS.items():
        if not isinstance(spec, PostTransferCalibrationSpec):
            raise ValueError("Post-transfer calibration registry values must be specs.")
        if key != spec.key:
            raise ValueError(
                "Post-transfer calibration registry key/spec mismatch: "
                f"{key!r} != {spec.key!r}."
            )
        live_specs.append(spec)

    payload: dict[str, object] = {
        "artifact_kind": _POLICY_ARTIFACT_KIND,
        "schema_version": _POLICY_SCHEMA_VERSION,
        "scope": {
            "reference": "asec_origin_clone_0",
            "recipient": "acs_origin_clone_0",
            "mutable": "caller_supplied_target_cells",
            "provenance_masks": "caller_supplied_no_internal_inference",
            "constraint_masks": "caller_supplied_hash_bound",
            "value_dtype": "float64_byte_contract",
            "zero_weight_rows": "byte_exact",
        },
        "quantiles": list(_QUANTILES),
        "carrier_selection": {
            "match_reference": "weighted_positive_prevalence_nearest_prefix",
            "removal_order": "positive_amount_descending_then_entity_id",
            "addition_order": "entity_id",
            "equal_distance": "lower_mass",
        },
        "amount_mapping": {
            "leg": "positive",
            "recipient_rank": "weighted_full_recipient_positive_upper_cdf",
            "inverse_cdf": "left",
            "exact_quantile_anchors": list(_QUANTILES),
            "infeasible_anchor_handling": "frame_owner_fail_closed",
            "output_support": "reference_positive_values_only",
        },
        "targets": [
            asdict(spec) for spec in sorted(live_specs, key=lambda item: item.key)
        ],
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def post_transfer_calibration_spec(
    *,
    entity: str,
    family: str,
    target: str,
) -> PostTransferCalibrationSpec:
    """Resolve one exact declared calibration spec."""

    key = f"{entity}/{family}/{target}"
    try:
        return POST_TRANSFER_CALIBRATION_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"No post-transfer calibration is declared for {key}."
        ) from exc


def post_transfer_calibration_spec_for_target(
    *,
    entity: str,
    target: str,
) -> PostTransferCalibrationSpec:
    """Resolve an entity/target only when its declared family is unambiguous."""

    matches = [
        spec
        for spec in POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.entity == entity and spec.target == target
    ]
    if len(matches) != 1:
        raise ValueError(
            "Post-transfer calibration entity/target lookup must resolve exactly "
            f"one spec; {entity}/{target} resolved {len(matches)}."
        )
    return matches[0]


def _require_declared_calibration_spec(
    spec: PostTransferCalibrationSpec,
) -> None:
    """Reject caller-constructed specs outside the live authority registry."""

    declared = POST_TRANSFER_CALIBRATION_SPECS.get(spec.key)
    if declared != spec:
        raise ValueError(
            "Post-transfer calibration spec is not the exact live declared "
            f"policy entry for {spec.key}."
        )


def _aligned_bool_mask(
    values: object,
    *,
    size: int,
    label: str,
) -> np.ndarray:
    mask = np.asarray(values)
    if mask.ndim != 1 or mask.shape != (size,) or mask.dtype.kind != "b":
        raise ValueError(
            f"{label} must be a one-dimensional boolean mask of length {size}."
        )
    return mask.astype(bool, copy=True)


def _numeric_values(values: object, *, label: str) -> np.ndarray:
    series = pd.Series(values, copy=False)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if numeric.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    return numeric


def _float64_values(values: object, *, label: str) -> np.ndarray:
    """Return an exact float64 vector or reject a lossy representation change."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if array.dtype != np.dtype(np.float64):
        raise ValueError(
            f"{label} must have exact float64 dtype so byte invariants are "
            f"meaningful; got {array.dtype}."
        )
    return array.copy()


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _stable_id_order(entity_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
    selected = entity_ids[positions]
    try:
        order = np.argsort(selected, kind="stable")
    except TypeError:
        tokens = np.asarray(
            [
                f"{type(value).__module__}.{type(value).__qualname__}:"
                f"{_json_scalar(value)!r}"
                for value in selected
            ],
            dtype=object,
        )
        order = np.argsort(tokens, kind="stable")
    return positions[order]


def _descending_value_then_id_order(
    values: np.ndarray,
    entity_ids: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    by_id = _stable_id_order(entity_ids, positions)
    return by_id[np.argsort(-values[by_id], kind="stable")]


def _prefix_schedule(
    ordered_positions: np.ndarray,
    weights: np.ndarray,
) -> _PrefixSchedule:
    positions = np.asarray(ordered_positions, dtype=np.int64).copy()
    cumulative = np.empty(len(positions) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(weights[positions], dtype=np.float64, out=cumulative[1:])
    positions.setflags(write=False)
    cumulative.setflags(write=False)
    return _PrefixSchedule(
        ordered_positions=positions,
        cumulative_mass=cumulative,
    )


def _nearest_prefix(
    schedule: _PrefixSchedule,
    target_mass: float,
) -> tuple[np.ndarray, float, dict[str, object]]:
    ordered_positions = schedule.ordered_positions
    cumulative = schedule.cumulative_mass
    requested = max(0.0, target_mass)
    # cumulative is ascending, so np.argmin implements the declared lower-mass
    # tie break when two adjacent prefixes are equally close.
    take = int(np.argmin(np.abs(cumulative - requested)))
    lower_index = max(
        0,
        min(
            int(np.searchsorted(cumulative, requested, side="right")) - 1,
            len(cumulative) - 1,
        ),
    )
    upper_index = min(lower_index + 1, len(cumulative) - 1)
    chosen_mass = float(cumulative[take])
    audit: dict[str, object] = {
        "requested_prefix_mass": float(requested),
        "candidate_rows": int(len(ordered_positions)),
        "candidate_mass": float(cumulative[-1]),
        "chosen_prefix_rows": take,
        "chosen_prefix_mass": chosen_mass,
        "lower_prefix_mass": float(cumulative[lower_index]),
        "upper_prefix_mass": float(cumulative[upper_index]),
    }
    return ordered_positions[:take], chosen_mass, audit


def _weighted_inverse_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] = _QUANTILES,
) -> np.ndarray | None:
    included = (values > 0.0) & (weights > 0.0)
    if not included.any():
        return None
    positions = np.flatnonzero(included)
    order = positions[np.argsort(values[positions], kind="stable")]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    cumulative /= cumulative[-1]
    indices = np.minimum(
        np.searchsorted(cumulative, np.asarray(probabilities), side="left"),
        len(order) - 1,
    )
    return values[order[indices]]


def _qed(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = np.abs(left) + np.abs(right)
    distances = np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(np.max(distances))


def _float_list(values: np.ndarray | None) -> list[float] | None:
    return None if values is None else [float(value) for value in values]


def _is_finite_number(value: object) -> bool:
    """Return whether a JSON scalar is a finite real without raising."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (OverflowError, TypeError, ValueError):
        return False


def _selected_bytes(values: np.ndarray, mask: np.ndarray) -> bytes:
    return np.ascontiguousarray(values[mask]).tobytes(order="C")


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes(order="C")).hexdigest()


def _values_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _map_positive_amounts(
    working: np.ndarray,
    *,
    original: np.ndarray,
    weights: np.ndarray,
    entity_ids: np.ndarray,
    reference_rows: np.ndarray,
    recipient_rows: np.ndarray,
    mutable_effective: np.ndarray,
    final_mutable_carriers: np.ndarray,
) -> dict[str, object]:
    donor_positions = np.flatnonzero(
        reference_rows & (weights > 0.0) & (original > 0.0)
    )
    donor_positions = _stable_id_order(entity_ids, donor_positions)
    donor_positions = donor_positions[
        np.argsort(original[donor_positions], kind="stable")
    ]
    donor_values = original[donor_positions]
    donor_weights = weights[donor_positions]
    donor_cumulative = np.cumsum(donor_weights, dtype=np.float64)
    donor_cumulative /= donor_cumulative[-1]

    recipient_carrier_rows = (
        recipient_rows & (weights > 0.0) & ((working > 0.0) | final_mutable_carriers)
    )
    recipient_positions = np.flatnonzero(recipient_carrier_rows)
    mutable_positions = np.flatnonzero(recipient_carrier_rows & mutable_effective)
    before_quantiles = _weighted_inverse_quantiles(
        original[recipient_rows], weights[recipient_rows]
    )
    reference_quantiles = _weighted_inverse_quantiles(
        original[reference_rows], weights[reference_rows]
    )
    anchor_conflicts: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    if recipient_positions.size:
        recipient_positions = _stable_id_order(entity_ids, recipient_positions)
        recipient_positions = recipient_positions[
            np.argsort(working[recipient_positions], kind="stable")
        ]
        recipient_cumulative = np.cumsum(weights[recipient_positions], dtype=np.float64)
        recipient_cumulative /= recipient_cumulative[-1]
        donor_indices = np.minimum(
            np.searchsorted(donor_cumulative, recipient_cumulative, side="left"),
            len(donor_positions) - 1,
        )
        mapped = donor_values[donor_indices]
        mutable_in_order = mutable_effective[recipient_positions]
        working[recipient_positions[mutable_in_order]] = mapped[mutable_in_order]

        # Resolve anchors against the complete recipient positive CDF, including
        # immutable carriers.  A sparse or immutable recipient can make two
        # distinct donor anchors compete for one row; that condition is recorded
        # explicitly and is rejected by the production owner.
        recipient_positions = _stable_id_order(entity_ids, recipient_positions)
        recipient_positions = recipient_positions[
            np.argsort(working[recipient_positions], kind="stable")
        ]
        recipient_cumulative = np.cumsum(weights[recipient_positions], dtype=np.float64)
        recipient_cumulative /= recipient_cumulative[-1]
        occupied: dict[int, tuple[float, float]] = {}
        for probability in _QUANTILES:
            recipient_rank = min(
                int(np.searchsorted(recipient_cumulative, probability, side="left")),
                len(recipient_positions) - 1,
            )
            donor_rank = min(
                int(np.searchsorted(donor_cumulative, probability, side="left")),
                len(donor_positions) - 1,
            )
            donor_value = float(donor_values[donor_rank])
            previous = occupied.get(recipient_rank)
            recipient_position = recipient_positions[recipient_rank]
            mutable_anchor = bool(mutable_effective[recipient_position])
            conflict_reason: str | None = None
            if previous is not None and previous[1] != donor_value:
                conflict_reason = "distinct_reference_values_share_recipient_row"
            elif not mutable_anchor and working[recipient_position] != donor_value:
                conflict_reason = "recipient_anchor_row_is_immutable"
            if conflict_reason is not None:
                anchor_conflicts.append(
                    {
                        "reason": conflict_reason,
                        "recipient_rank": recipient_rank,
                        "earlier_quantile": (None if previous is None else previous[0]),
                        "earlier_reference_value": (
                            None if previous is None else previous[1]
                        ),
                        "quantile": probability,
                        "reference_value": donor_value,
                    }
                )
            occupied[recipient_rank] = (probability, donor_value)
            if mutable_anchor:
                working[recipient_position] = donor_value
            anchor_rows.append(
                {
                    "quantile": probability,
                    "entity_id": _json_scalar(entity_ids[recipient_position]),
                    "recipient_rank": recipient_rank,
                    "reference_value": donor_value,
                    "mutable": mutable_anchor,
                }
            )

    after_quantiles = _weighted_inverse_quantiles(
        working[recipient_rows], weights[recipient_rows]
    )
    unanchored: list[float] = (
        list(_QUANTILES)
        if reference_quantiles is not None and after_quantiles is None
        else []
    )
    if reference_quantiles is not None and after_quantiles is not None:
        unanchored = [
            probability
            for probability, reference_value, recipient_value in zip(
                _QUANTILES,
                reference_quantiles,
                after_quantiles,
                strict=True,
            )
            if recipient_value != reference_value
        ]
    mapped_values = working[mutable_positions]
    support_violations = int(
        np.count_nonzero(~np.isin(mapped_values, donor_values, assume_unique=False))
    )
    return {
        "quantiles": list(_QUANTILES),
        "reference_quantiles": _float_list(reference_quantiles),
        "recipient_before_quantiles": _float_list(before_quantiles),
        "recipient_after_quantiles": _float_list(after_quantiles),
        "qed_before": _qed(reference_quantiles, before_quantiles),
        "qed_after": _qed(reference_quantiles, after_quantiles),
        "mapped_rows": int(mutable_positions.size),
        "anchor_rows": anchor_rows,
        "exact_anchor_count": len(_QUANTILES) - len(unanchored),
        "anchor_conflicts": anchor_conflicts,
        "unanchored_quantiles": unanchored,
        "donor_support_violations": support_violations,
        "status": (
            "exact"
            if not anchor_conflicts and not unanchored
            else "infeasible_exact_anchors"
        ),
    }


def calibrate_post_transfer_values(
    values: object,
    weights: object,
    entity_ids: object,
    *,
    spec: PostTransferCalibrationSpec,
    reference_rows: object,
    recipient_rows: object,
    mutable_rows: object,
    allowed_carrier_rows: object | None = None,
    addition_candidate_rows: object | None = None,
) -> PostTransferCalibrationResult:
    """Calibrate one positive transfer leg without reading provenance columns."""

    if not isinstance(spec, PostTransferCalibrationSpec):
        raise TypeError("spec must be a PostTransferCalibrationSpec.")
    _require_declared_calibration_spec(spec)
    if spec.special_constraint != "none" and (
        allowed_carrier_rows is None or addition_candidate_rows is None
    ):
        raise ValueError(
            f"Post-transfer calibration {spec.key} requires explicit "
            "allowed_carrier_rows and addition_candidate_rows for special "
            f"constraint {spec.special_constraint!r}."
        )
    original = _float64_values(values, label="values")
    size = len(original)
    numeric_weights = _numeric_values(weights, label="weights")
    ids = np.asarray(entity_ids)
    if len(numeric_weights) != size or ids.ndim != 1 or len(ids) != size:
        raise ValueError("values, weights, and entity_ids must align one-to-one.")
    if not np.isfinite(numeric_weights).all() or (numeric_weights < 0.0).any():
        raise ValueError(
            "Post-transfer calibration weights must be finite and nonnegative."
        )
    if pd.Series(ids).isna().any() or pd.Series(ids).duplicated().any():
        raise ValueError(
            "Post-transfer calibration entity_ids must be complete and unique."
        )

    reference = _aligned_bool_mask(reference_rows, size=size, label="reference_rows")
    recipient = _aligned_bool_mask(recipient_rows, size=size, label="recipient_rows")
    mutable = _aligned_bool_mask(mutable_rows, size=size, label="mutable_rows")
    if (reference & recipient).any():
        raise ValueError("reference_rows and recipient_rows must be disjoint.")
    if (mutable & ~recipient).any():
        raise ValueError("mutable_rows must be a subset of recipient_rows.")
    if not (reference & (numeric_weights > 0.0)).any():
        raise ValueError("Reference rows have no positive-weight support.")
    if not (recipient & (numeric_weights > 0.0)).any():
        raise ValueError("Recipient rows have no positive-weight support.")
    relevant = (reference | recipient) & (numeric_weights > 0.0)
    if not np.isfinite(original[relevant]).all():
        raise ValueError(
            "Positive-weight reference and recipient values must be finite."
        )
    if not (reference & (numeric_weights > 0.0) & (original > 0.0)).any():
        raise ValueError("Reference rows have no positive donor support.")

    mutable_effective = mutable & (numeric_weights > 0.0)
    allowed = (
        mutable.copy()
        if allowed_carrier_rows is None
        else _aligned_bool_mask(
            allowed_carrier_rows,
            size=size,
            label="allowed_carrier_rows",
        )
    )
    additions = (
        allowed.copy()
        if addition_candidate_rows is None
        else _aligned_bool_mask(
            addition_candidate_rows,
            size=size,
            label="addition_candidate_rows",
        )
    )
    for label, mask in (
        ("allowed_carrier_rows", allowed),
        ("addition_candidate_rows", additions),
    ):
        if (mask & ~(mutable & recipient)).any():
            raise ValueError(f"{label} must be a subset of mutable recipient rows.")
    if (additions & ~allowed).any():
        raise ValueError("addition_candidate_rows must be a subset of allowed rows.")

    protected = ~mutable_effective
    negative = original < 0.0
    negative_zero = (original == 0.0) & np.signbit(original)
    zero_weight = numeric_weights == 0.0
    protected_bytes = _selected_bytes(original, protected)
    negative_bytes = _selected_bytes(original, negative)
    negative_zero_bytes = _selected_bytes(original, negative_zero)
    zero_weight_bytes = _selected_bytes(original, zero_weight)

    working = original.copy()
    reference_total = float(numeric_weights[reference].sum())
    recipient_total = float(numeric_weights[recipient].sum())
    reference_positive_mass = float(numeric_weights[reference & (original > 0.0)].sum())
    reference_share = reference_positive_mass / reference_total
    target_positive_mass = reference_share * recipient_total
    before_positive = recipient & (original > 0.0)
    before_positive_mass = float(numeric_weights[before_positive].sum())
    before_carriers = before_positive.copy()
    removed = np.zeros(size, dtype=bool)
    added = np.zeros(size, dtype=bool)
    disallowed = np.zeros(size, dtype=bool)
    capacity_limited = False
    capacity_receipt: dict[str, object] | None = None
    selection_receipt: dict[str, object] | None = None

    if spec.carrier_mode == "match_reference":
        disallowed = mutable_effective & (working > 0.0) & ~allowed
        working[disallowed] = 0.0
        removed |= disallowed
        fixed_positive = recipient & (working > 0.0) & ~mutable_effective
        fixed_mass = float(numeric_weights[fixed_positive].sum())
        allowed_positive = mutable_effective & allowed & (working > 0.0)
        allowed_positive_order = _descending_value_then_id_order(
            working,
            ids,
            np.flatnonzero(allowed_positive),
        )
        allowed_positive_schedule = _prefix_schedule(
            allowed_positive_order,
            numeric_weights,
        )
        allowed_positive_mass = float(allowed_positive_schedule.cumulative_mass[-1])
        zero_candidates = (
            mutable_effective
            & allowed
            & additions
            & (working == 0.0)
            & ~np.signbit(working)
        )
        addition_candidate_order = _stable_id_order(
            ids,
            np.flatnonzero(zero_candidates),
        )
        addition_candidate_schedule = _prefix_schedule(
            addition_candidate_order,
            numeric_weights,
        )
        addition_candidate_mass = float(addition_candidate_schedule.cumulative_mass[-1])
        minimum_attainable_mass = fixed_mass
        maximum_attainable_mass = (
            fixed_mass + allowed_positive_mass + addition_candidate_mass
        )
        capacity_limited = bool(
            target_positive_mass < minimum_attainable_mass
            or target_positive_mass > maximum_attainable_mass
        )
        desired_mutable_mass = max(0.0, target_positive_mass - fixed_mass)
        if desired_mutable_mass <= allowed_positive_mass:
            retained, retained_mass, prefix_audit = _nearest_prefix(
                allowed_positive_schedule,
                desired_mutable_mass,
            )
            retained_mask = np.zeros(size, dtype=bool)
            retained_mask[retained] = True
            clear = allowed_positive & ~retained_mask
            working[clear] = 0.0
            removed |= clear
            selection_receipt = {
                "action": "retain_positive_prefix",
                "base_positive_mass": fixed_mass,
                "selected_rows": int(len(retained)),
                "selected_mass": retained_mass,
                **prefix_audit,
            }
        else:
            needed = desired_mutable_mass - allowed_positive_mass
            selected, selected_mass, prefix_audit = _nearest_prefix(
                addition_candidate_schedule,
                needed,
            )
            added[selected] = True
            selection_receipt = {
                "action": "add_zero_prefix",
                "base_positive_mass": fixed_mass + allowed_positive_mass,
                "selected_rows": int(len(selected)),
                "selected_mass": selected_mass,
                **prefix_audit,
            }
        capacity_receipt = {
            "fixed_positive_rows": int(fixed_positive.sum()),
            "fixed_positive_mass": fixed_mass,
            "allowed_positive_rows_before": int(allowed_positive.sum()),
            "allowed_positive_mass_before": allowed_positive_mass,
            "addition_candidate_rows": int(zero_candidates.sum()),
            "addition_candidate_mass": addition_candidate_mass,
            "minimum_attainable_mass": minimum_attainable_mass,
            "maximum_attainable_mass": maximum_attainable_mass,
            "target_within_attainable_interval": not capacity_limited,
        }

    final_mutable_carriers = mutable_effective & ((working > 0.0) | added)
    amount_receipt = _map_positive_amounts(
        working,
        original=original,
        weights=numeric_weights,
        entity_ids=ids,
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_effective=mutable_effective,
        final_mutable_carriers=final_mutable_carriers,
    )
    after_positive = recipient & (working > 0.0)
    after_positive_mass = float(numeric_weights[after_positive].sum())
    after_share = after_positive_mass / recipient_total

    if spec.carrier_mode == "match_reference":
        assert capacity_receipt is not None
        assert selection_receipt is not None
        minimum = float(capacity_receipt["minimum_attainable_mass"])
        maximum = float(capacity_receipt["maximum_attainable_mass"])
        if target_positive_mass < minimum:
            saturated = np.isclose(after_positive_mass, minimum, rtol=1e-12, atol=1e-12)
        elif target_positive_mass > maximum:
            saturated = np.isclose(after_positive_mass, maximum, rtol=1e-12, atol=1e-12)
        else:
            saturated = True
        capacity_receipt["capacity_boundary_saturated"] = bool(saturated)
        if not saturated:
            raise ValueError(
                "Post-transfer carrier calibration did not saturate its "
                "attainable-mass boundary."
            )

    immutable_ok = _selected_bytes(working, protected) == protected_bytes
    negative_ok = _selected_bytes(working, negative) == negative_bytes
    negative_zero_ok = _selected_bytes(working, negative_zero) == negative_zero_bytes
    zero_weight_ok = _selected_bytes(working, zero_weight) == zero_weight_bytes
    preserve_carriers = bool(
        np.array_equal(before_carriers[recipient], after_positive[recipient])
    )
    allowed_violations = int(
        (after_positive & mutable_effective & ~allowed).sum()
        if spec.carrier_mode == "match_reference"
        else 0
    )
    exact_quantile_anchors = bool(
        amount_receipt["status"] == "exact"
        and amount_receipt["exact_anchor_count"] == len(_QUANTILES)
        and not amount_receipt["anchor_conflicts"]
        and not amount_receipt["unanchored_quantiles"]
    )
    if not all(
        (
            immutable_ok,
            negative_ok,
            negative_zero_ok,
            zero_weight_ok,
            amount_receipt["donor_support_violations"] == 0,
            allowed_violations == 0,
        )
    ):
        raise ValueError("Post-transfer calibration violated a byte/support invariant.")
    if spec.carrier_mode == "preserve_recipient" and not preserve_carriers:
        raise ValueError("Preserve-recipient calibration changed carrier membership.")

    policy = post_transfer_calibration_policy_identity()
    receipt: dict[str, object] = {
        "artifact_kind": "microcosm_us_post_transfer_calibration_receipt",
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "verification_contract": _receipt_verification_contract(),
        "policy_sha256": policy["sha256"],
        "spec": asdict(spec),
        "scope": {
            "rows": size,
            "reference_rows": int(reference.sum()),
            "recipient_rows": int(recipient.sum()),
            "mutable_rows": int(mutable.sum()),
            "effective_mutable_rows": int(mutable_effective.sum()),
            "reference_rows_sha256": _mask_sha256(reference),
            "recipient_rows_sha256": _mask_sha256(recipient),
            "mutable_rows_sha256": _mask_sha256(mutable),
            "allowed_carrier_rows": int(allowed.sum()),
            "allowed_carrier_rows_sha256": _mask_sha256(allowed),
            "allowed_carrier_rows_mode": (
                "default_mutable" if allowed_carrier_rows is None else "caller_supplied"
            ),
            "addition_candidate_rows": int(additions.sum()),
            "addition_candidate_rows_sha256": _mask_sha256(additions),
            "addition_candidate_rows_mode": (
                "default_allowed"
                if addition_candidate_rows is None
                else "caller_supplied"
            ),
            "entity_ids_sha256": _canonical_sha256(
                [_json_scalar(value) for value in ids.tolist()]
            ),
            "input_values_sha256": _values_sha256(original),
            "output_values_sha256": _values_sha256(working),
        },
        "weights": {
            "sha256": _values_sha256(numeric_weights),
            "reference_total": reference_total,
            "recipient_total": recipient_total,
        },
        "carrier": {
            "mode": spec.carrier_mode,
            "reference_positive_mass": reference_positive_mass,
            "reference_positive_share": reference_share,
            "target_positive_mass": target_positive_mass,
            "before_positive_mass": before_positive_mass,
            "before_positive_share": before_positive_mass / recipient_total,
            "after_positive_mass": after_positive_mass,
            "after_positive_share": after_share,
            "residual_after_minus_target": after_positive_mass - target_positive_mass,
            "absolute_residual": abs(after_positive_mass - target_positive_mass),
            "removed_rows": int(removed.sum()),
            "added_rows": int(added.sum()),
            "disallowed_cleared_rows": int(disallowed.sum()),
            "capacity_limited": capacity_limited,
            "capacity": capacity_receipt,
            "selection": selection_receipt,
        },
        "amount": amount_receipt,
        "invariants": {
            "immutable_bytes_preserved": immutable_ok,
            "negative_bytes_preserved": negative_ok,
            "negative_zero_bytes_preserved": negative_zero_ok,
            "zero_weight_bytes_preserved": zero_weight_ok,
            "preserve_carriers": preserve_carriers,
            "allowed_carrier_violations": allowed_violations,
            "exact_quantile_anchors": exact_quantile_anchors,
        },
    }
    return PostTransferCalibrationResult(
        values=working,
        receipt={**receipt, "sha256": _canonical_sha256(receipt)},
    )


def validate_post_transfer_calibration_receipt(
    receipt: Mapping[str, object],
    *,
    spec: PostTransferCalibrationSpec,
    boundary: str,
    expected_policy_sha256: str | None = None,
    expected_scope: Mapping[str, object] | None = None,
    expected_weights_sha256: str | None = None,
    require_exact_anchors: bool = True,
) -> None:
    """Validate a receipt's schema and generation-time audit relationships.

    Terminal callers must separately replay the fields named by
    ``verification_contract.terminal_live_receipt_paths`` against a live final
    frame.  Pre-calibration transition claims cannot be reconstructed from that
    frame and are authenticated only by their generation context and enclosing
    execution authority.
    """

    if not isinstance(spec, PostTransferCalibrationSpec):
        raise TypeError("spec must be a PostTransferCalibrationSpec.")
    _require_declared_calibration_spec(spec)
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{boundary}: post-transfer calibration receipt is absent.")
    payload = dict(receipt)
    observed_sha256 = payload.pop("sha256", None)
    if observed_sha256 != _canonical_sha256(payload):
        raise ValueError(
            f"{boundary}: post-transfer calibration receipt digest is invalid."
        )
    policy_sha256 = (
        post_transfer_calibration_policy_identity()["sha256"]
        if expected_policy_sha256 is None
        else expected_policy_sha256
    )
    if (
        receipt.get("artifact_kind") != "microcosm_us_post_transfer_calibration_receipt"
        or receipt.get("schema_version") != _RECEIPT_SCHEMA_VERSION
        or receipt.get("verification_contract") != _receipt_verification_contract()
        or receipt.get("policy_sha256") != policy_sha256
        or receipt.get("spec") != asdict(spec)
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration policy/spec binding is invalid."
        )
    scope = receipt.get("scope")
    required_scope_digests = {
        "reference_rows_sha256",
        "recipient_rows_sha256",
        "mutable_rows_sha256",
        "allowed_carrier_rows_sha256",
        "addition_candidate_rows_sha256",
        "entity_ids_sha256",
        "input_values_sha256",
        "output_values_sha256",
    }
    required_scope_counts = {
        "rows",
        "reference_rows",
        "recipient_rows",
        "mutable_rows",
        "effective_mutable_rows",
        "allowed_carrier_rows",
        "addition_candidate_rows",
    }
    required_scope_modes = {
        "allowed_carrier_rows_mode",
        "addition_candidate_rows_mode",
    }
    if not isinstance(scope, Mapping) or not (
        required_scope_digests | required_scope_counts | required_scope_modes
    ).issubset(scope):
        raise ValueError(
            f"{boundary}: post-transfer calibration scope evidence is incomplete."
        )
    if any(
        not isinstance(scope.get(key), int)
        or isinstance(scope.get(key), bool)
        or scope[key] < 0
        for key in required_scope_counts
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration scope counts are invalid."
        )
    if (
        scope["reference_rows"] > scope["rows"]
        or scope["recipient_rows"] > scope["rows"]
        or scope["mutable_rows"] > scope["recipient_rows"]
        or scope["effective_mutable_rows"] > scope["mutable_rows"]
        or scope["allowed_carrier_rows"] > scope["mutable_rows"]
        or scope["addition_candidate_rows"] > scope["allowed_carrier_rows"]
        or scope["allowed_carrier_rows_mode"]
        not in {"default_mutable", "caller_supplied"}
        or scope["addition_candidate_rows_mode"]
        not in {"default_allowed", "caller_supplied"}
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration scope relationships are invalid."
        )
    if spec.special_constraint != "none" and (
        scope["allowed_carrier_rows_mode"] != "caller_supplied"
        or scope["addition_candidate_rows_mode"] != "caller_supplied"
    ):
        raise ValueError(
            f"{boundary}: post-transfer special-constraint masks were not "
            "explicitly supplied."
        )
    digest_values = [scope.get(key) for key in required_scope_digests]
    weights_receipt = receipt.get("weights")
    if not isinstance(weights_receipt, Mapping):
        raise ValueError(
            f"{boundary}: post-transfer calibration weight evidence is absent."
        )
    digest_values.append(weights_receipt.get("sha256"))
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in digest_values
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration context digest is invalid."
        )
    for total_key in ("reference_total", "recipient_total"):
        value = weights_receipt.get(total_key)
        if not _is_finite_number(value) or value <= 0.0:
            raise ValueError(
                f"{boundary}: post-transfer calibration weight totals are invalid."
            )
    if expected_scope is not None and any(
        scope.get(key) != value for key, value in expected_scope.items()
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration scope does not match the "
            "expected live context."
        )
    if expected_weights_sha256 is not None and (
        weights_receipt.get("sha256") != expected_weights_sha256
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration weights do not match the "
            "expected live context."
        )
    invariants = receipt.get("invariants")
    required_true = {
        "immutable_bytes_preserved",
        "negative_bytes_preserved",
        "negative_zero_bytes_preserved",
        "zero_weight_bytes_preserved",
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not True for key in required_true
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration byte invariants are invalid."
        )
    if invariants.get("allowed_carrier_violations") != 0:
        raise ValueError(
            f"{boundary}: post-transfer calibration carrier constraint failed."
        )
    if spec.carrier_mode == "preserve_recipient" and (
        invariants.get("preserve_carriers") is not True
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration changed frozen carriers."
        )
    carrier = receipt.get("carrier")
    required_carrier_masses = {
        "reference_positive_mass",
        "reference_positive_share",
        "target_positive_mass",
        "before_positive_mass",
        "before_positive_share",
        "after_positive_mass",
        "after_positive_share",
        "residual_after_minus_target",
        "absolute_residual",
    }
    required_carrier_counts = {
        "removed_rows",
        "added_rows",
        "disallowed_cleared_rows",
    }
    expected_carrier_keys = (
        required_carrier_masses
        | required_carrier_counts
        | {"mode", "capacity_limited", "capacity", "selection"}
    )
    if (
        not isinstance(carrier, Mapping)
        or carrier.get("mode") != spec.carrier_mode
        or set(carrier) != expected_carrier_keys
        or not isinstance(carrier.get("capacity_limited"), bool)
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration carrier evidence is incomplete."
        )
    if any(
        not _is_finite_number(carrier.get(key)) for key in required_carrier_masses
    ) or any(
        not isinstance(carrier.get(key), int)
        or isinstance(carrier.get(key), bool)
        or carrier[key] < 0
        for key in required_carrier_counts
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration carrier values are invalid."
        )
    reference_total = float(weights_receipt["reference_total"])
    recipient_total = float(weights_receipt["recipient_total"])
    expected_carrier_values = {
        "reference_positive_share": (
            float(carrier["reference_positive_mass"]) / reference_total
        ),
        "target_positive_mass": (
            float(carrier["reference_positive_share"]) * recipient_total
        ),
        "before_positive_share": (
            float(carrier["before_positive_mass"]) / recipient_total
        ),
        "after_positive_share": (
            float(carrier["after_positive_mass"]) / recipient_total
        ),
        "residual_after_minus_target": (
            float(carrier["after_positive_mass"])
            - float(carrier["target_positive_mass"])
        ),
        "absolute_residual": abs(float(carrier["residual_after_minus_target"])),
    }
    if any(
        not np.isclose(
            float(carrier[key]),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        for key, expected in expected_carrier_values.items()
    ) or (
        float(carrier["reference_positive_mass"]) <= 0.0
        or float(carrier["reference_positive_mass"]) > reference_total
        or not 0.0 < float(carrier["reference_positive_share"]) <= 1.0
        or float(carrier["target_positive_mass"]) <= 0.0
        or float(carrier["before_positive_mass"]) > recipient_total
        or float(carrier["after_positive_mass"]) > recipient_total
        or not 0.0 <= float(carrier["before_positive_share"]) <= 1.0
        or not 0.0 <= float(carrier["after_positive_share"]) <= 1.0
        or carrier["removed_rows"] > scope["effective_mutable_rows"]
        or carrier["added_rows"] > scope["addition_candidate_rows"]
        or carrier["disallowed_cleared_rows"] > carrier["removed_rows"]
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration carrier relationships are invalid."
        )

    capacity = carrier.get("capacity")
    selection = carrier.get("selection")
    if spec.carrier_mode == "preserve_recipient":
        if (
            capacity is not None
            or selection is not None
            or carrier["capacity_limited"] is not False
            or any(carrier[key] != 0 for key in required_carrier_counts)
            or not np.isclose(
                float(carrier["after_positive_mass"]),
                float(carrier["before_positive_mass"]),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"{boundary}: preserve-recipient carrier proof is invalid."
            )
    else:
        capacity_count_keys = {
            "fixed_positive_rows",
            "allowed_positive_rows_before",
            "addition_candidate_rows",
        }
        capacity_mass_keys = {
            "fixed_positive_mass",
            "allowed_positive_mass_before",
            "addition_candidate_mass",
            "minimum_attainable_mass",
            "maximum_attainable_mass",
        }
        expected_capacity_keys = (
            capacity_count_keys
            | capacity_mass_keys
            | {
                "target_within_attainable_interval",
                "capacity_boundary_saturated",
            }
        )
        selection_count_keys = {
            "selected_rows",
            "candidate_rows",
            "chosen_prefix_rows",
        }
        selection_mass_keys = {
            "base_positive_mass",
            "selected_mass",
            "requested_prefix_mass",
            "candidate_mass",
            "chosen_prefix_mass",
            "lower_prefix_mass",
            "upper_prefix_mass",
        }
        expected_selection_keys = (
            selection_count_keys | selection_mass_keys | {"action"}
        )
        if (
            not isinstance(capacity, Mapping)
            or set(capacity) != expected_capacity_keys
            or not isinstance(selection, Mapping)
            or set(selection) != expected_selection_keys
            or any(
                not isinstance(capacity.get(key), int)
                or isinstance(capacity.get(key), bool)
                or capacity[key] < 0
                for key in capacity_count_keys
            )
            or any(
                not _is_finite_number(capacity.get(key)) or capacity[key] < 0.0
                for key in capacity_mass_keys
            )
            or not isinstance(capacity.get("target_within_attainable_interval"), bool)
            or capacity.get("capacity_boundary_saturated") is not True
            or any(
                not isinstance(selection.get(key), int)
                or isinstance(selection.get(key), bool)
                or selection[key] < 0
                for key in selection_count_keys
            )
            or any(
                not _is_finite_number(selection.get(key)) or selection[key] < 0.0
                for key in selection_mass_keys
            )
            or selection.get("action")
            not in {"retain_positive_prefix", "add_zero_prefix"}
        ):
            raise ValueError(
                f"{boundary}: match-reference carrier capacity proof is invalid."
            )
        fixed_mass = float(capacity["fixed_positive_mass"])
        allowed_mass = float(capacity["allowed_positive_mass_before"])
        addition_mass = float(capacity["addition_candidate_mass"])
        minimum = float(capacity["minimum_attainable_mass"])
        maximum = float(capacity["maximum_attainable_mass"])
        target = float(carrier["target_positive_mass"])
        expected_limited = target < minimum or target > maximum
        action = selection["action"]
        desired_mutable_mass = max(0.0, target - fixed_mass)
        expected_action = (
            "retain_positive_prefix"
            if desired_mutable_mass <= allowed_mass
            else "add_zero_prefix"
        )
        expected_candidate_rows = (
            capacity["allowed_positive_rows_before"]
            if action == "retain_positive_prefix"
            else capacity["addition_candidate_rows"]
        )
        expected_candidate_mass = (
            allowed_mass if action == "retain_positive_prefix" else addition_mass
        )
        expected_base = (
            fixed_mass
            if action == "retain_positive_prefix"
            else (fixed_mass + allowed_mass)
        )
        expected_requested = (
            max(0.0, target - fixed_mass)
            if action == "retain_positive_prefix"
            else max(0.0, target - fixed_mass - allowed_mass)
        )
        lower = float(selection["lower_prefix_mass"])
        upper = float(selection["upper_prefix_mass"])
        expected_chosen = (
            lower
            if abs(lower - expected_requested) <= abs(upper - expected_requested)
            else upper
        )
        after_mass = float(carrier["after_positive_mass"])
        boundary_mass = minimum if target < minimum else maximum
        if (
            not np.isclose(minimum, fixed_mass, rtol=1e-12, atol=1e-12)
            or action != expected_action
            or capacity["fixed_positive_rows"]
            > scope["recipient_rows"] - scope["effective_mutable_rows"]
            or capacity["allowed_positive_rows_before"] > scope["allowed_carrier_rows"]
            or capacity["addition_candidate_rows"] > scope["addition_candidate_rows"]
            or sum(capacity[key] for key in capacity_count_keys)
            > scope["recipient_rows"]
            or fixed_mass > recipient_total
            or allowed_mass > recipient_total
            or addition_mass > recipient_total
            or not np.isclose(
                maximum,
                fixed_mass + allowed_mass + addition_mass,
                rtol=1e-12,
                atol=1e-12,
            )
            or maximum > recipient_total
            or carrier["capacity_limited"] is not expected_limited
            or capacity["target_within_attainable_interval"] is expected_limited
            or selection["candidate_rows"] != expected_candidate_rows
            or selection["selected_rows"] != selection["chosen_prefix_rows"]
            or selection["selected_rows"] > selection["candidate_rows"]
            or not np.isclose(
                float(selection["candidate_mass"]),
                expected_candidate_mass,
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.isclose(
                float(selection["base_positive_mass"]),
                expected_base,
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.isclose(
                float(selection["requested_prefix_mass"]),
                expected_requested,
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.isclose(
                float(selection["selected_mass"]),
                float(selection["chosen_prefix_mass"]),
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.isclose(
                float(selection["chosen_prefix_mass"]),
                expected_chosen,
                rtol=1e-12,
                atol=1e-12,
            )
            or not (0.0 <= lower <= upper <= expected_candidate_mass)
            or not np.isclose(
                after_mass,
                expected_base + float(selection["selected_mass"]),
                rtol=1e-12,
                atol=1e-12,
            )
            or (
                expected_limited
                and not np.isclose(
                    after_mass,
                    boundary_mass,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        ):
            raise ValueError(
                f"{boundary}: match-reference carrier capacity relationships "
                "are invalid."
            )

    amount = receipt.get("amount")
    expected_amount_keys = {
        "quantiles",
        "reference_quantiles",
        "recipient_before_quantiles",
        "recipient_after_quantiles",
        "qed_before",
        "qed_after",
        "mapped_rows",
        "anchor_rows",
        "exact_anchor_count",
        "anchor_conflicts",
        "unanchored_quantiles",
        "donor_support_violations",
        "status",
    }
    if not isinstance(amount, Mapping) or set(amount) != expected_amount_keys:
        raise ValueError(
            f"{boundary}: post-transfer calibration amount schema is invalid."
        )
    reference_quantiles = amount.get("reference_quantiles")
    before_quantiles = amount.get("recipient_before_quantiles")
    after_quantiles = amount.get("recipient_after_quantiles")
    quantile_vectors = (reference_quantiles, after_quantiles)
    if (
        amount.get("quantiles") != list(_QUANTILES)
        or any(
            not isinstance(vector, list)
            or len(vector) != len(_QUANTILES)
            or any(not _is_finite_number(value) or value <= 0.0 for value in vector)
            for vector in quantile_vectors
        )
        or (
            before_quantiles is not None
            and (
                not isinstance(before_quantiles, list)
                or len(before_quantiles) != len(_QUANTILES)
                or any(
                    not _is_finite_number(value) or value <= 0.0
                    for value in before_quantiles
                )
            )
        )
        or not isinstance(amount.get("mapped_rows"), int)
        or isinstance(amount.get("mapped_rows"), bool)
        or amount["mapped_rows"] < 0
        or amount["mapped_rows"] > scope["effective_mutable_rows"]
        or not isinstance(amount.get("exact_anchor_count"), int)
        or isinstance(amount.get("exact_anchor_count"), bool)
        or not 0 <= amount["exact_anchor_count"] <= len(_QUANTILES)
        or not isinstance(amount.get("donor_support_violations"), int)
        or isinstance(amount.get("donor_support_violations"), bool)
        or amount["donor_support_violations"] != 0
        or not isinstance(amount.get("anchor_rows"), list)
        or not isinstance(amount.get("anchor_conflicts"), list)
        or not isinstance(amount.get("unanchored_quantiles"), list)
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration donor support is invalid."
        )
    for label, observed, expected in (
        (
            "qed_before",
            amount.get("qed_before"),
            _qed(
                np.asarray(reference_quantiles, dtype=np.float64),
                (
                    None
                    if before_quantiles is None
                    else np.asarray(before_quantiles, dtype=np.float64)
                ),
            ),
        ),
        (
            "qed_after",
            amount.get("qed_after"),
            _qed(
                np.asarray(reference_quantiles, dtype=np.float64),
                np.asarray(after_quantiles, dtype=np.float64),
            ),
        ),
    ):
        if expected is None:
            valid = observed is None
        else:
            valid = _is_finite_number(observed) and np.isclose(
                float(observed), expected, rtol=1e-12, atol=1e-12
            )
        if not valid:
            raise ValueError(
                f"{boundary}: post-transfer calibration {label} is invalid."
            )
    unanchored = amount["unanchored_quantiles"]
    conflicts = amount["anchor_conflicts"]
    expected_status = (
        "exact" if not conflicts and not unanchored else "infeasible_exact_anchors"
    )
    anchor_keys = {
        "quantile",
        "entity_id",
        "recipient_rank",
        "reference_value",
        "mutable",
    }
    conflict_keys = {
        "reason",
        "recipient_rank",
        "earlier_quantile",
        "earlier_reference_value",
        "quantile",
        "reference_value",
    }
    if (
        len(amount["anchor_rows"]) != len(_QUANTILES)
        or any(
            not isinstance(row, Mapping)
            or set(row) != anchor_keys
            or row.get("quantile") != probability
            or not isinstance(row.get("recipient_rank"), int)
            or isinstance(row.get("recipient_rank"), bool)
            or row["recipient_rank"] < 0
            or row["recipient_rank"] >= scope["recipient_rows"]
            or row.get("entity_id") is None
            or not _is_finite_number(row.get("reference_value"))
            or row["reference_value"] <= 0.0
            or not np.isclose(
                float(row["reference_value"]),
                float(reference_quantiles[index]),
                rtol=1e-12,
                atol=1e-12,
            )
            or not isinstance(row.get("mutable"), bool)
            for index, (row, probability) in enumerate(
                zip(amount["anchor_rows"], _QUANTILES, strict=True)
            )
        )
        or any(
            not isinstance(conflict, Mapping)
            or set(conflict) != conflict_keys
            or conflict.get("reason")
            not in {
                "distinct_reference_values_share_recipient_row",
                "recipient_anchor_row_is_immutable",
            }
            or not isinstance(conflict.get("recipient_rank"), int)
            or isinstance(conflict.get("recipient_rank"), bool)
            or conflict["recipient_rank"] < 0
            or conflict["recipient_rank"] >= scope["recipient_rows"]
            or not _is_finite_number(conflict.get("quantile"))
            or conflict["quantile"] not in _QUANTILES
            or (
                conflict.get("earlier_quantile") is not None
                and (
                    not _is_finite_number(conflict["earlier_quantile"])
                    or conflict["earlier_quantile"] not in _QUANTILES
                )
            )
            or not _is_finite_number(conflict.get("reference_value"))
            or conflict["reference_value"] <= 0.0
            or (
                conflict.get("earlier_reference_value") is not None
                and (
                    not _is_finite_number(conflict["earlier_reference_value"])
                    or conflict["earlier_reference_value"] <= 0.0
                )
            )
            for conflict in conflicts
        )
        or any(
            not _is_finite_number(value) or value not in _QUANTILES
            for value in unanchored
        )
        or len({float(value) for value in unanchored}) != len(unanchored)
        or amount["exact_anchor_count"] != len(_QUANTILES) - len(unanchored)
        or amount.get("status") != expected_status
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration anchor evidence is invalid."
        )
    if require_exact_anchors and (
        invariants.get("exact_quantile_anchors") is not True
        or amount.get("status") != "exact"
        or amount.get("exact_anchor_count") != len(_QUANTILES)
        or amount.get("anchor_conflicts") != []
        or amount.get("unanchored_quantiles") != []
    ):
        raise ValueError(
            f"{boundary}: post-transfer calibration exact quantile anchors are "
            "infeasible."
        )


def apply_post_transfer_calibration(
    frame: Frame,
    *,
    entity: str,
    target: str,
    reference_rows: object,
    recipient_rows: object,
    mutable_rows: object,
    allowed_carrier_rows: object | None = None,
    addition_candidate_rows: object | None = None,
    family: str | None = None,
) -> PostTransferCalibrationFrameResult:
    """Apply one calibration using Frame-resolved weights and entity IDs."""

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    spec = (
        post_transfer_calibration_spec_for_target(entity=entity, target=target)
        if family is None
        else post_transfer_calibration_spec(
            entity=entity,
            family=family,
            target=target,
        )
    )
    table = frame.table(entity)
    if target not in table:
        raise ValueError(
            f"Post-transfer calibration target {entity}/{target} is absent."
        )
    id_column = frame.schema.entity_id_column(entity)
    resolved_weights = frame.resolve_weights(entity)
    result = calibrate_post_transfer_values(
        table[target].to_numpy(copy=True),
        resolved_weights.values,
        table[id_column].to_numpy(copy=False),
        spec=spec,
        reference_rows=reference_rows,
        recipient_rows=recipient_rows,
        mutable_rows=mutable_rows,
        allowed_carrier_rows=allowed_carrier_rows,
        addition_candidate_rows=addition_candidate_rows,
    )
    mutable_mask = _aligned_bool_mask(
        mutable_rows,
        size=len(table),
        label="mutable_rows",
    )
    reference_mask = _aligned_bool_mask(
        reference_rows,
        size=len(table),
        label="reference_rows",
    )
    recipient_mask = _aligned_bool_mask(
        recipient_rows,
        size=len(table),
        label="recipient_rows",
    )
    allowed_mask = (
        mutable_mask
        if allowed_carrier_rows is None
        else _aligned_bool_mask(
            allowed_carrier_rows,
            size=len(table),
            label="allowed_carrier_rows",
        )
    )
    addition_mask = (
        allowed_mask
        if addition_candidate_rows is None
        else _aligned_bool_mask(
            addition_candidate_rows,
            size=len(table),
            label="addition_candidate_rows",
        )
    )
    original_numeric = _float64_values(table[target].to_numpy(copy=False), label=target)
    changed_rows = mutable_mask & (
        original_numeric.view(np.uint64) != result.values.view(np.uint64)
    )
    tables = {name: frame.table(name) for name in frame.entities}
    calibrated = table.copy(deep=True)
    calibrated.loc[changed_rows, target] = result.values[changed_rows]
    tables[entity] = calibrated
    tables.update({name: frame.link(name) for name in frame.links})
    output = Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    receipt = {
        **result.receipt,
        "scope": {
            **result.receipt["scope"],
            "applied_changed_rows": int(changed_rows.sum()),
        },
        "weights": {
            **result.receipt["weights"],
            "kind": resolved_weights.kind.value,
        },
    }
    receipt_without_hash = dict(receipt)
    receipt_without_hash.pop("sha256", None)
    receipt["sha256"] = _canonical_sha256(receipt_without_hash)
    validate_post_transfer_calibration_receipt(
        receipt,
        spec=spec,
        boundary=f"Frame post-transfer calibration {spec.key}",
        expected_scope={
            "reference_rows_sha256": _mask_sha256(reference_mask),
            "recipient_rows_sha256": _mask_sha256(recipient_mask),
            "mutable_rows_sha256": _mask_sha256(mutable_mask),
            "allowed_carrier_rows_sha256": _mask_sha256(allowed_mask),
            "addition_candidate_rows_sha256": _mask_sha256(addition_mask),
            "entity_ids_sha256": _canonical_sha256(
                [
                    _json_scalar(value)
                    for value in table[id_column].to_numpy(copy=False).tolist()
                ]
            ),
            "input_values_sha256": _values_sha256(original_numeric),
            "output_values_sha256": _values_sha256(result.values),
        },
        expected_weights_sha256=_values_sha256(
            _numeric_values(resolved_weights.values, label="weights")
        ),
    )
    return PostTransferCalibrationFrameResult(frame=output, receipt=receipt)
