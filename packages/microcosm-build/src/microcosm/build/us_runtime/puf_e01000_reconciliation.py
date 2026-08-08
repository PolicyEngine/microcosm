"""Audit-only reconciliation of PUF E01000 and Schedule-D capital gains.

E01000 is not a PolicyEngine carrier in the processed PUF or output frame.
This module therefore records the source concept alongside the signed
P22250/P23250 carrier without creating, reconstructing, or changing a column.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.puf_aggregate_records import (
    PufAggregateDisaggregationSpec,
    load_default_puf_aggregate_disaggregation_spec,
)

__all__ = [
    "PUF_E01000_RECONCILIATION_SCHEMA_VERSION",
    "build_puf_e01000_reconciliation_basis",
    "finalize_puf_e01000_reconciliation",
    "puf_capital_gains_joint_metrics",
    "puf_processed_capital_gains_stage",
    "puf_raw_e01000_stage",
]

PUF_E01000_RECONCILIATION_SCHEMA_VERSION = 1
PUF_E01000_SOURCE_YEAR = 2015
PUF_E01000_SOURCE_COLUMNS = (
    "RECID",
    "S006",
    "E01000",
    "P22250",
    "P23250",
)
PUF_SCHEDULE_D_JOINT_COLUMNS = (
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
)
_METRIC_KEYS = (
    "record_count",
    "total_weight",
    "positive_carrier_count",
    "positive_carrier_weight",
    "weighted_signed_mass",
    "weighted_positive_mass",
)


def puf_capital_gains_joint_metrics(
    donor: pd.DataFrame,
    *,
    mask: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    """Measure signed ST+LT after summing the two legs at donor-row grain."""

    required = {"weight", *PUF_SCHEDULE_D_JOINT_COLUMNS}
    missing = sorted(required - set(donor.columns))
    if missing:
        raise ValueError(
            f"PUF capital-gains reconciliation donor is missing columns: {missing}."
        )
    weights = _finite_numeric(donor["weight"], label="donor weight")
    short_term = _finite_numeric(
        donor[PUF_SCHEDULE_D_JOINT_COLUMNS[0]],
        label=PUF_SCHEDULE_D_JOINT_COLUMNS[0],
    )
    long_term = _finite_numeric(
        donor[PUF_SCHEDULE_D_JOINT_COLUMNS[1]],
        label=PUF_SCHEDULE_D_JOINT_COLUMNS[1],
    )
    selected = _validated_mask(mask, len(donor))
    return _weighted_metric(
        short_term[selected] + long_term[selected],
        weights[selected],
    )


def puf_processed_capital_gains_stage(
    donor: pd.DataFrame,
    *,
    spec: PufAggregateDisaggregationSpec | None = None,
) -> dict[str, object]:
    """Classify the already-processed donor without re-disaggregating it."""

    e01000_columns = _e01000_columns(donor.columns)
    if e01000_columns:
        raise ValueError(
            "Processed PUF donor unexpectedly carries E01000; the "
            f"reconciliation is audit-only, found {e01000_columns}."
        )
    if "tax_unit_id" not in donor:
        raise ValueError(
            "PUF capital-gains reconciliation donor is missing tax_unit_id."
        )
    resolved_spec = spec or load_default_puf_aggregate_disaggregation_spec()
    ids_numeric = _finite_numeric(donor["tax_unit_id"], label="tax_unit_id")
    if not np.equal(ids_numeric, np.floor(ids_numeric)).all():
        raise ValueError("PUF reconciliation tax_unit_id values must be integers.")
    ids = ids_numeric.astype(np.int64)
    retained_aggregate_ids = sorted(
        set(ids).intersection(resolved_spec.aggregate_recids)
    )
    if retained_aggregate_ids:
        raise ValueError(
            "Processed PUF still contains source aggregate RECIDs; refusing to "
            f"re-disaggregate or mislabel them: {retained_aggregate_ids}."
        )
    synthetic = ids >= resolved_spec.synthetic_recid_start
    regular = ~synthetic
    return {
        "status": "observed_upstream_replacement",
        "classification": {
            "regular": (
                f"tax_unit_id < {resolved_spec.synthetic_recid_start}; source "
                "aggregate RECIDs absent"
            ),
            "synthetic": (f"tax_unit_id >= {resolved_spec.synthetic_recid_start}"),
        },
        "source_aggregate_record_ids": list(resolved_spec.aggregate_recids),
        "source_aggregate_record_ids_absent": True,
        "synthetic_recid_start": int(resolved_spec.synthetic_recid_start),
        "synthetic_tail_support_eligible": bool(
            resolved_spec.synthetic_tail_support_eligible
        ),
        "regular": puf_capital_gains_joint_metrics(donor, mask=regular),
        "synthetic": puf_capital_gains_joint_metrics(donor, mask=synthetic),
        "all": puf_capital_gains_joint_metrics(donor),
    }


def puf_raw_e01000_stage(
    source_puf_csv: Path,
    *,
    spec: PufAggregateDisaggregationSpec | None = None,
) -> dict[str, object]:
    """Measure raw TY2015 E01000 and row-joint P22250/P23250 concepts."""

    return _raw_source_stage(
        Path(source_puf_csv),
        spec=spec or load_default_puf_aggregate_disaggregation_spec(),
    )


def build_puf_e01000_reconciliation_basis(
    source_puf_csv: Path,
    donor: pd.DataFrame,
    *,
    processed_before_screen: Mapping[str, object],
    mortgage_screen: Mapping[str, object],
    target_year: int,
    source_sha256: str | None = None,
    spec: PufAggregateDisaggregationSpec | None = None,
) -> dict[str, object]:
    """Build the source-through-donor portion of the audit receipt."""

    resolved_spec = spec or load_default_puf_aggregate_disaggregation_spec()
    source_path = Path(source_puf_csv)
    raw_source = puf_raw_e01000_stage(source_path, spec=resolved_spec)
    donor_stage = puf_processed_capital_gains_stage(donor, spec=resolved_spec)
    donor_stage["status"] = "post_field_local_mortgage_screen"
    replacement = deepcopy(dict(processed_before_screen))
    screen = deepcopy(dict(mortgage_screen))
    _validate_processed_stage(replacement, label="synthetic replacement")
    _validate_processed_stage(donor_stage, label="donor")
    _assert_metric_groups_equal(
        replacement,
        donor_stage,
        label="mortgage field quarantine",
    )
    screen["all_records"] = {
        "before": _project_metric(replacement["all"]),
        "after": _project_metric(donor_stage["all"]),
        "difference": _metric_difference(
            replacement["all"],
            donor_stage["all"],
        ),
    }
    _validate_screen_preservation(screen)

    e01000_positive = float(raw_source["all"]["e01000"]["weighted_positive_mass"])
    schedule_d_positive = float(
        raw_source["all"]["p22250_plus_p23250"]["weighted_positive_mass"]
    )
    difference = schedule_d_positive - e01000_positive
    if e01000_positive <= 0.0:
        raise ValueError("Raw PUF E01000 positive mass must be positive.")

    return {
        "artifact_kind": "populace_puf_e01000_capital_gains_reconciliation",
        "schema_version": PUF_E01000_RECONCILIATION_SCHEMA_VERSION,
        "source": {
            "tax_year": PUF_E01000_SOURCE_YEAR,
            "path": str(source_path.resolve()),
            "sha256": source_sha256 or _sha256(source_path),
            "weight_column": "S006",
            "weight_scale": 0.01,
            "aggregate_record_ids": list(resolved_spec.aggregate_recids),
        },
        "target": {
            "period": int(target_year),
            "status": "uprated_processed_puf_and_frame",
        },
        "carrier": {
            "source_column": "E01000",
            "source_status_after_raw": "audit_only_not_carried",
            "donor_column": None,
            "frame_column": None,
            "materialized": False,
            "carrier_changed": False,
            "donor_column_absence_verified": True,
            "frame_column_absence_verified": False,
            "frame_entities_checked": [],
        },
        "concepts": {
            "e01000": {
                "source_columns": ["E01000"],
                "aggregation": "row_value",
            },
            "schedule_d_joint": {
                "source_columns": ["P22250", "P23250"],
                "processed_columns": list(PUF_SCHEDULE_D_JOINT_COLUMNS),
                "aggregation": "sum_signed_legs_at_row_grain_before_positive_clip",
                "legs_scaled_independently": False,
            },
        },
        "concept_divergence": {
            "basis": "weighted_positive_mass",
            "denominator": "e01000_weighted_positive_mass",
            "e01000_weighted_positive_mass": e01000_positive,
            "p22250_plus_p23250_weighted_positive_mass": schedule_d_positive,
            "difference": difference,
            "ratio": difference / e01000_positive,
            "percent": 100.0 * difference / e01000_positive,
            "interpretation": ("expected_source_concept_divergence_not_stage_loss"),
        },
        "reconciliation_policy": {
            "raw_to_processed": ("cross_period_uprated_not_a_conservation_identity"),
            "aggregate_replacement": (
                "observe_processed_artifact_without_re_disaggregation"
            ),
            "mortgage_screen": "schedule_d_joint_carrier_must_be_unchanged",
            "e01000": "audit_only_no_carrier_change",
        },
        "stages": {
            "raw_source": raw_source,
            "synthetic_replacement": {
                "period": int(target_year),
                "e01000_status": "not_carried",
                **replacement,
            },
            "mortgage_screen": {
                "period": int(target_year),
                "e01000_status": "not_carried",
                **screen,
            },
            "donor": {
                "period": int(target_year),
                "e01000_status": "not_carried",
                **donor_stage,
            },
        },
    }


def finalize_puf_e01000_reconciliation(
    basis: Mapping[str, object],
    capital_gains_tail_transfer: Mapping[str, object],
    *,
    frame_columns: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    """Attach frame receipts from the mass-conserving tail-transfer stage."""

    receipt = deepcopy(dict(basis))
    if receipt.get("schema_version") != PUF_E01000_RECONCILIATION_SCHEMA_VERSION:
        raise ValueError("Unsupported PUF E01000 reconciliation basis schema.")
    stages = receipt.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("PUF E01000 reconciliation basis is missing stages.")
    target = receipt.get("target")
    if not isinstance(target, Mapping) or "period" not in target:
        raise ValueError("PUF E01000 reconciliation basis is missing target period.")
    if not frame_columns:
        raise ValueError(
            "PUF E01000 reconciliation requires a final-frame column inventory."
        )
    frame_matches = {
        str(entity): matches
        for entity, columns in frame_columns.items()
        if (matches := _e01000_columns(columns))
    }
    if frame_matches:
        raise ValueError(
            "Final frame unexpectedly carries E01000; the reconciliation is "
            f"audit-only, found {frame_matches}."
        )
    carrier = receipt.get("carrier")
    if not isinstance(carrier, dict):
        raise ValueError("PUF E01000 reconciliation basis is missing carrier policy.")
    carrier["frame_column_absence_verified"] = True
    carrier["frame_entities_checked"] = sorted(str(entity) for entity in frame_columns)
    distributions = capital_gains_tail_transfer.get("tail_distribution_receipts")
    if not isinstance(distributions, Mapping):
        raise ValueError(
            "PUF capital-gains tail metadata is missing distribution receipts."
        )
    before = distributions.get("frame_before_stage")
    after = distributions.get("frame_after_stage")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ValueError(
            "PUF capital-gains tail metadata is missing frame before/after receipts."
        )
    stages["frame"] = {
        "period": int(target["period"]),
        "e01000_status": "not_carried",
        "schedule_d_joint_columns": list(PUF_SCHEDULE_D_JOINT_COLUMNS),
        "before_tail_transfer": _project_metric(before),
        "after_tail_transfer": _project_metric(after),
    }
    return receipt


def _raw_source_stage(
    source_path: Path,
    *,
    spec: PufAggregateDisaggregationSpec,
) -> dict[str, object]:
    try:
        raw = pd.read_csv(source_path, usecols=list(PUF_E01000_SOURCE_COLUMNS))
    except ValueError as exc:
        raise ValueError(
            "Raw PUF E01000 reconciliation source must contain "
            f"{list(PUF_E01000_SOURCE_COLUMNS)}."
        ) from exc
    columns = {
        column: _finite_numeric(raw[column], label=f"raw PUF {column}")
        for column in PUF_E01000_SOURCE_COLUMNS
    }
    ids_numeric = columns["RECID"]
    if not np.equal(ids_numeric, np.floor(ids_numeric)).all():
        raise ValueError("Raw PUF RECID values must be integers.")
    ids = ids_numeric.astype(np.int64)
    # Match the PUF's documented hundredths-of-a-return encoding literally.
    # Division also pins the diagnostic receipt's measured binary reduction.
    weights = columns["S006"] / 100.0
    if (weights < 0.0).any() or weights.sum() <= 0.0:
        raise ValueError("Raw PUF S006 design weights must be nonnegative.")
    observed_counts = {
        recid: int((ids == recid).sum()) for recid in spec.aggregate_recids
    }
    invalid_counts = {
        recid: count for recid, count in observed_counts.items() if count != 1
    }
    if invalid_counts:
        raise ValueError(
            "Raw PUF must contain each declared aggregate RECID exactly once; "
            f"got {invalid_counts}."
        )
    aggregate = np.isin(ids, spec.aggregate_recids)
    e01000 = columns["E01000"]
    schedule_d = columns["P22250"] + columns["P23250"]

    def cohort(mask: np.ndarray) -> dict[str, object]:
        return {
            "e01000": _weighted_metric(e01000[mask], weights[mask]),
            "p22250_plus_p23250": _weighted_metric(
                schedule_d[mask],
                weights[mask],
            ),
        }

    return {
        "period": PUF_E01000_SOURCE_YEAR,
        "regular": cohort(~aggregate),
        "aggregate": cohort(aggregate),
        "all": cohort(np.ones(len(raw), dtype=bool)),
    }


def _weighted_metric(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("PUF reconciliation values and weights must align.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("PUF reconciliation values and weights must be finite.")
    if (weights < 0.0).any():
        raise ValueError("PUF reconciliation weights must be nonnegative.")
    positive = values > 0.0
    return {
        "record_count": int(len(values)),
        "total_weight": float(weights.sum()),
        "positive_carrier_count": int(positive.sum()),
        "positive_carrier_weight": float(weights[positive].sum()),
        "weighted_signed_mass": float(np.dot(values, weights)),
        "weighted_positive_mass": float(np.dot(np.maximum(values, 0.0), weights)),
    }


def _finite_numeric(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain only finite numeric values.")
    return numeric


def _validated_mask(mask: Sequence[bool] | None, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 1 or len(selected) != length:
        raise ValueError(
            "PUF capital-gains reconciliation mask must align with donor rows."
        )
    return selected


def _e01000_columns(columns: Sequence[object]) -> list[str]:
    return sorted(
        str(column) for column in columns if str(column).strip().upper() == "E01000"
    )


def _validate_processed_stage(stage: Mapping[str, object], *, label: str) -> None:
    required = {
        "status",
        "classification",
        "source_aggregate_record_ids",
        "source_aggregate_record_ids_absent",
        "synthetic_recid_start",
        "synthetic_tail_support_eligible",
        "regular",
        "synthetic",
        "all",
    }
    missing = sorted(required - set(stage))
    if missing:
        raise ValueError(f"PUF {label} receipt is missing fields: {missing}.")
    for cohort in ("regular", "synthetic", "all"):
        _project_metric(stage[cohort])


def _assert_metric_groups_equal(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    label: str,
) -> None:
    for cohort in ("regular", "synthetic", "all"):
        before_metric = _project_metric(before[cohort])
        after_metric = _project_metric(after[cohort])
        for key in _METRIC_KEYS:
            if before_metric[key] != after_metric[key]:
                raise AssertionError(
                    f"PUF {label} changed {cohort} {key}: "
                    f"{before_metric[key]} != {after_metric[key]}."
                )


def _validate_screen_preservation(screen: Mapping[str, object]) -> None:
    for name in ("capital_gains_preserved", "all_records"):
        preserved = screen.get(name)
        if not isinstance(preserved, Mapping):
            raise ValueError(f"PUF mortgage screen receipt is missing {name}.")
        before = _project_metric(preserved.get("before"))
        after = _project_metric(preserved.get("after"))
        difference = _project_metric(preserved.get("difference"))
        for key in _METRIC_KEYS:
            expected_difference = after[key] - before[key]
            if difference[key] != expected_difference or expected_difference != 0:
                raise AssertionError(
                    "PUF mortgage screen changed the Schedule-D joint carrier "
                    f"{name} {key}: before={before[key]}, after={after[key]}, "
                    f"difference={difference[key]}."
                )


def _project_metric(metric: object) -> dict[str, float | int]:
    if not isinstance(metric, Mapping):
        raise ValueError("PUF capital-gains reconciliation metric must be a mapping.")
    missing = sorted(set(_METRIC_KEYS) - set(metric))
    if missing:
        raise ValueError(
            f"PUF capital-gains reconciliation metric is missing: {missing}."
        )
    return {key: metric[key] for key in _METRIC_KEYS}


def _metric_difference(
    before: object,
    after: object,
) -> dict[str, float | int]:
    before_metric = _project_metric(before)
    after_metric = _project_metric(after)
    return {key: after_metric[key] - before_metric[key] for key in _METRIC_KEYS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
