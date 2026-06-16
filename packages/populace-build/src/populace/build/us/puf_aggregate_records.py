"""Spec-declared disaggregation for IRS PUF aggregate disclosure rows.

The IRS PUF replaces a small set of extreme-tail returns with four aggregate
rows. This stage replaces those disclosure rows with synthetic donor templates
drawn from the same AGI buckets, while deriving all amount totals from the raw
PUF aggregate rows at runtime.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "AGGREGATE_RECIDS",
    "SYNTHETIC_RECID_START",
    "PufAggregateDisaggregationSpec",
    "compute_aggregate_eligibility_scores",
    "disaggregate_puf_aggregate_records",
    "load_default_puf_aggregate_disaggregation_spec",
]

AGGREGATE_RECIDS = (999996, 999997, 999998, 999999)
SYNTHETIC_RECID_START = 1_000_000

_AMOUNT_COLUMN_PATTERN = re.compile(r"^(?:[EPT]\d+|S\d{5})$")
_STRUCTURAL_COLUMNS = ("MARS", "XTOT", "DSI", "EIC")
_MAX_AGI_DOMINANCE = 0.20
_SELECTION_POWER = 24
_NUMERIC_TOL = 1e-9
_WEIGHTED_TOTAL_ABS_TOL = 1e-4
_WEIGHTED_TOTAL_REL_TOL = 1e-10
_SPEC_ALLOWED_KEYS = {
    "enabled",
    "forbes_top_tail",
    "source",
    "aggregate_recids",
    "synthetic_recid_start",
    "synthetic_tail_support_eligible",
    "screened_fields",
    "buckets",
}
_BUCKET_ALLOWED_KEYS = {
    "description",
    "agi_lower",
    "agi_upper",
    "synthetic_agi_upper",
}


@dataclass(frozen=True)
class AggregateBucketSpec:
    """One raw aggregate-record AGI bucket."""

    recid: int
    description: str
    agi_lower: float | None
    agi_upper: float | None
    synthetic_agi_upper: float | None

    def contains(self, values: pd.Series) -> pd.Series:
        mask = pd.Series(True, index=values.index)
        if self.agi_lower is not None:
            mask &= values >= self.agi_lower
        if self.agi_upper is not None:
            mask &= values < self.agi_upper
        return mask


@dataclass(frozen=True)
class PufAggregateDisaggregationSpec:
    """Declarative configuration for the aggregate-row transform."""

    enabled: bool
    forbes_top_tail: bool
    source: str
    aggregate_recids: tuple[int, ...]
    synthetic_recid_start: int
    screened_fields: tuple[str, ...]
    synthetic_tail_support_eligible: bool
    buckets: dict[int, AggregateBucketSpec]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PufAggregateDisaggregationSpec:
        _reject_unknown_keys(raw, allowed=_SPEC_ALLOWED_KEYS, context="aggregate spec")
        buckets = {
            int(recid): cls._bucket_from_dict(recid, spec)
            for recid, spec in raw["buckets"].items()
        }
        aggregate_recids = tuple(int(recid) for recid in raw["aggregate_recids"])
        missing = sorted(set(aggregate_recids) - set(buckets))
        if missing:
            raise ValueError(f"Aggregate spec missing bucket metadata for {missing}.")
        spec = cls(
            enabled=bool(raw["enabled"]),
            forbes_top_tail=bool(raw["forbes_top_tail"]),
            source=str(raw["source"]),
            aggregate_recids=aggregate_recids,
            synthetic_recid_start=int(raw["synthetic_recid_start"]),
            screened_fields=tuple(str(field) for field in raw["screened_fields"]),
            synthetic_tail_support_eligible=bool(
                raw["synthetic_tail_support_eligible"]
            ),
            buckets=buckets,
        )
        spec.validate()
        return spec

    @staticmethod
    def _bucket_from_dict(recid: str, raw: dict[str, Any]) -> AggregateBucketSpec:
        _reject_unknown_keys(
            raw,
            allowed=_BUCKET_ALLOWED_KEYS,
            context=f"aggregate bucket {recid}",
        )
        return AggregateBucketSpec(
            recid=int(recid),
            description=str(raw["description"]),
            agi_lower=_optional_float(raw.get("agi_lower")),
            agi_upper=_optional_float(raw.get("agi_upper")),
            synthetic_agi_upper=_optional_float(raw.get("synthetic_agi_upper")),
        )

    def validate(self) -> None:
        if self.forbes_top_tail:
            raise ValueError(
                "Forbes top-tail synthesis is intentionally not enabled in "
                "the aggregate-record-only disaggregation spec."
            )
        if not self.aggregate_recids:
            raise ValueError("Aggregate disaggregation spec has no aggregate RECIDs.")
        if self.synthetic_recid_start <= max(self.aggregate_recids):
            raise ValueError(
                "synthetic_recid_start must exceed the raw aggregate RECIDs."
            )
        if not self.screened_fields:
            raise ValueError("Aggregate disaggregation spec has no screened fields.")


def load_default_puf_aggregate_disaggregation_spec() -> PufAggregateDisaggregationSpec:
    """Load the packaged aggregate-row transform declaration."""

    path = files("populace.build.us") / "puf_aggregate_record_disaggregation.json"
    return PufAggregateDisaggregationSpec.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def disaggregate_puf_aggregate_records(
    puf: pd.DataFrame,
    *,
    seed: int = 42,
    spec: PufAggregateDisaggregationSpec | None = None,
) -> pd.DataFrame:
    """Replace raw PUF aggregate rows with calibrated synthetic donors."""

    spec = spec or load_default_puf_aggregate_disaggregation_spec()
    spec.validate()
    if not spec.enabled:
        return puf.copy()
    _require_columns(puf, ["RECID", "S006", "E00100"])

    aggregate_recids = list(spec.aggregate_recids)
    aggregate_mask = puf["RECID"].isin(aggregate_recids)
    if int(aggregate_mask.sum()) == 0:
        return puf.copy()
    if int(aggregate_mask.sum()) != len(aggregate_recids):
        found = sorted(puf.loc[aggregate_mask, "RECID"].astype(int).tolist())
        raise ValueError(
            "PUF aggregate disaggregation expected all aggregate RECIDs "
            f"{aggregate_recids}, found {found}."
        )

    rng = np.random.default_rng(seed)
    amount_columns = _get_amount_columns(puf.columns)
    aggregate_rows = puf[aggregate_mask].copy().set_index("RECID")
    regular = puf[~aggregate_mask].copy()
    donor_scores = compute_aggregate_eligibility_scores(
        regular,
        screened_fields=list(spec.screened_fields),
    )

    pieces: list[pd.DataFrame] = []
    next_recid = spec.synthetic_recid_start
    for recid in aggregate_recids:
        synthetic = _disaggregate_bucket(
            recid=recid,
            row=aggregate_rows.loc[recid],
            regular=regular,
            amount_columns=amount_columns,
            donor_scores=donor_scores,
            next_recid=next_recid,
            rng=rng,
            spec=spec,
        )
        next_recid += len(synthetic)
        pieces.append(synthetic[puf.columns])

    synthetic_df = pd.concat(pieces, ignore_index=True)
    return pd.concat([regular, synthetic_df], ignore_index=True)


def compute_aggregate_eligibility_scores(
    df: pd.DataFrame,
    *,
    screened_fields: list[str] | None = None,
    reference_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Score records on how similar they are to disclosure-pooled returns."""

    spec = load_default_puf_aggregate_disaggregation_spec()
    fields = screened_fields or list(spec.screened_fields)
    reference = df if reference_df is None else reference_df
    present_fields = [field for field in fields if field in df.columns]
    if not present_fields:
        return pd.Series(0.0, index=df.index, dtype=float)

    max_scores = np.zeros(len(df), dtype=float)
    for field in present_fields:
        values = pd.to_numeric(df[field], errors="coerce").fillna(0.0)
        reference_values = pd.to_numeric(
            reference[field],
            errors="coerce",
        ).fillna(0.0)
        field_scores = np.zeros(len(df), dtype=float)

        positive = values > 0
        reference_positive = np.sort(reference_values[reference_values > 0].to_numpy())
        if positive.any() and len(reference_positive) > 0:
            positive_scores = np.searchsorted(
                reference_positive,
                values[positive].to_numpy(),
                side="right",
            ) / len(reference_positive)
            field_scores[positive.to_numpy()] = positive_scores

        negative = values < 0
        reference_negative = np.sort(
            (-reference_values[reference_values < 0]).to_numpy()
        )
        if negative.any() and len(reference_negative) > 0:
            negative_scores = np.searchsorted(
                reference_negative,
                (-values[negative]).to_numpy(),
                side="right",
            ) / len(reference_negative)
            field_scores[negative.to_numpy()] = np.maximum(
                field_scores[negative.to_numpy()],
                negative_scores,
            )
        max_scores = np.maximum(max_scores, field_scores)

    return pd.Series(max_scores, index=df.index, dtype=float)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _finite_float(value: object, *, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    result = float(numeric)
    return result if np.isfinite(result) else default


def _reject_unknown_keys(
    raw: dict[str, Any],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError(f"{context} has unsupported keys: {unexpected}.")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"PUF aggregate disaggregation missing columns {missing}.")


def _choose_n_synthetic(pop_weight: float) -> int:
    return int(min(40, max(20, round(pop_weight / 10))))


def _assign_s006_values(
    total_s006: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive when assigning synthetic S006 values.")
    if total_s006 < n:
        raise ValueError(
            "total_s006 must be at least n to assign positive integer S006 values."
        )

    base = total_s006 // n
    weights = np.full(n, base, dtype=int)
    remainder = total_s006 - int(weights.sum())

    if remainder > 0:
        weights[rng.choice(n, size=remainder, replace=False)] += 1
    return weights


def _get_amount_columns(columns: pd.Index | list[str]) -> list[str]:
    return [column for column in columns if _AMOUNT_COLUMN_PATTERN.match(column)]


def _get_bucket_targets(row: pd.Series) -> tuple[float, float, float]:
    pop_weight = _finite_float(row["S006"]) / 100.0
    target_mean_agi = _finite_float(row["E00100"])
    return pop_weight, target_mean_agi, pop_weight * target_mean_agi


def _get_donor_bucket(
    regular: pd.DataFrame,
    recid: int,
    spec: PufAggregateDisaggregationSpec,
) -> pd.DataFrame:
    bucket = regular[
        spec.buckets[recid].contains(
            pd.to_numeric(regular["E00100"], errors="coerce").fillna(0.0)
        )
    ].copy()
    if bucket.empty:
        return regular.copy()
    return bucket


def _coerce_amount_columns(
    selected: pd.DataFrame,
    amount_columns: list[str],
) -> pd.DataFrame:
    coerced = selected.copy()
    for column in amount_columns:
        if column in coerced.columns:
            coerced[column] = (
                pd.to_numeric(coerced[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
    return coerced


def _project_weighted_sum_to_bounds(
    values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    projected = np.clip(values.astype(float), lower, upper)

    for _ in range(max_iter):
        residual = float(target_total - np.dot(projected, weights))
        if abs(residual) <= 1e-6:
            return projected

        slack = upper - projected if residual > 0 else projected - lower
        free = slack > _NUMERIC_TOL
        if not free.any():
            break

        basis = np.abs(projected[free])
        if basis.sum() <= _NUMERIC_TOL:
            basis = np.ones(free.sum(), dtype=float)

        denom = float(np.dot(weights[free], basis))
        if denom <= _NUMERIC_TOL:
            basis = np.ones(free.sum(), dtype=float)
            denom = float(weights[free].sum())

        delta = residual * basis / denom
        if residual > 0:
            delta = np.minimum(delta, slack[free])
        else:
            delta = -np.minimum(-delta, slack[free])

        projected[free] += delta
        projected = np.clip(projected, lower, upper)

    residual = float(target_total - np.dot(projected, weights))
    if abs(residual) > 1e-6:
        slack = upper - projected if residual > 0 else projected - lower
        free_indices = np.where(slack > _NUMERIC_TOL)[0]
        if len(free_indices) > 0:
            best = free_indices[np.argmax(slack[free_indices] * weights[free_indices])]
            projected[best] = np.clip(
                projected[best] + residual / weights[best],
                lower[best],
                upper[best],
            )
    return projected


def _allocate_weighted_values(
    base_values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
    lower: np.ndarray | float | None = None,
    upper: np.ndarray | float | None = None,
) -> np.ndarray:
    base_values = np.asarray(base_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(base_values)

    if abs(target_total) <= 1e-6:
        return np.zeros(n, dtype=float)

    if target_total > 0 and np.any(base_values > 0):
        active = base_values > 0
    elif target_total < 0 and np.any(base_values < 0):
        active = base_values < 0
    elif np.any(np.abs(base_values) > _NUMERIC_TOL):
        active = np.abs(base_values) > _NUMERIC_TOL
    else:
        active = np.ones(n, dtype=bool)

    allocated = np.zeros(n, dtype=float)
    magnitudes = np.abs(base_values[active])
    if magnitudes.sum() <= _NUMERIC_TOL:
        magnitudes = np.ones(active.sum(), dtype=float)

    denom = float(np.dot(weights[active], magnitudes))
    if denom <= _NUMERIC_TOL:
        magnitudes = np.ones(active.sum(), dtype=float)
        denom = float(weights[active].sum())
    allocated[active] = np.sign(target_total) * magnitudes * abs(target_total) / denom

    if lower is None and upper is None:
        return allocated

    lower_array = (
        np.full(n, -np.inf, dtype=float)
        if lower is None
        else np.full(n, float(lower), dtype=float)
        if np.isscalar(lower)
        else np.asarray(lower, dtype=float)
    )
    upper_array = (
        np.full(n, np.inf, dtype=float)
        if upper is None
        else np.full(n, float(upper), dtype=float)
        if np.isscalar(upper)
        else np.asarray(upper, dtype=float)
    )
    return _project_weighted_sum_to_bounds(
        allocated,
        weights,
        target_total,
        lower_array,
        upper_array,
    )


def _assert_weighted_total_matches(
    *,
    column: str,
    recid: int,
    values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
) -> None:
    if not np.isfinite(target_total):
        raise ValueError(
            f"PUF aggregate disaggregation target for {column} RECID {recid} "
            f"is not finite: {target_total}."
        )
    achieved = float(np.dot(np.asarray(values, dtype=float), weights))
    residual = target_total - achieved
    if not np.isfinite(achieved) or not np.isfinite(residual):
        raise ValueError(
            f"PUF aggregate disaggregation achieved non-finite {column} for "
            f"RECID {recid}: target={target_total}, achieved={achieved}, "
            f"residual={residual}."
        )
    tolerance = max(
        _WEIGHTED_TOTAL_ABS_TOL,
        abs(target_total) * _WEIGHTED_TOTAL_REL_TOL,
    )
    if abs(residual) > tolerance:
        raise ValueError(
            f"PUF aggregate disaggregation could not preserve {column} for "
            f"RECID {recid}: target={target_total}, achieved={achieved}, "
            f"residual={residual}."
        )


def _allocate_agi_values(
    donor_agi: np.ndarray,
    weights: np.ndarray,
    recid: int,
    target_total: float,
    spec: PufAggregateDisaggregationSpec,
) -> np.ndarray:
    donor_agi = np.asarray(donor_agi, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(donor_agi)
    bucket = spec.buckets[recid]
    dominance_cap = _MAX_AGI_DOMINANCE * abs(target_total) / weights

    if bucket.agi_upper == 0:
        lower = -dominance_cap
        upper = np.zeros(n, dtype=float)
    else:
        bucket_lower = float(bucket.agi_lower or 0.0)
        bucket_upper = (
            bucket.synthetic_agi_upper
            if bucket.agi_upper is None
            else float(bucket.agi_upper)
        )
        if bucket_upper is None:
            bucket_upper = np.inf
        lower = np.full(n, max(bucket_lower, 0.0), dtype=float)
        upper = np.minimum(np.full(n, bucket_upper, dtype=float), dominance_cap)

    allocated = _allocate_weighted_values(
        base_values=np.abs(donor_agi),
        weights=weights,
        target_total=target_total,
        lower=lower,
        upper=upper,
    )
    _assert_weighted_total_matches(
        column="E00100",
        recid=recid,
        values=allocated,
        weights=weights,
        target_total=target_total,
    )
    return allocated


def _selection_probabilities(
    donor_bucket: pd.DataFrame,
    donor_scores: pd.Series,
    target_mean_agi: float,
) -> np.ndarray:
    scores = donor_scores.loc[donor_bucket.index].to_numpy(dtype=float)
    score_mass = np.clip(scores, 1e-6, None) ** _SELECTION_POWER

    donor_abs_agi = np.abs(donor_bucket["E00100"].to_numpy(dtype=float))
    target_abs_agi = max(abs(float(target_mean_agi)), 1.0)
    agi_distance = np.abs(np.log1p(donor_abs_agi) - np.log1p(target_abs_agi))
    agi_mass = 1.0 / (1.0 + agi_distance)

    probabilities = score_mass * np.sqrt(agi_mass)
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        probabilities = np.ones(len(donor_bucket), dtype=float)
    return probabilities / probabilities.sum()


def _sample_bucket_donors(
    donor_bucket: pd.DataFrame,
    donor_scores: pd.Series,
    target_mean_agi: float,
    n_synthetic: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    probabilities = _selection_probabilities(
        donor_bucket=donor_bucket,
        donor_scores=donor_scores,
        target_mean_agi=target_mean_agi,
    )
    selected_index = rng.choice(
        donor_bucket.index.to_numpy(),
        size=n_synthetic,
        replace=len(donor_bucket) < n_synthetic,
        p=probabilities,
    )
    return donor_bucket.loc[selected_index].reset_index(drop=True).copy()


def _apply_structural_templates(
    synthetic: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    for column in _STRUCTURAL_COLUMNS:
        if column in synthetic.columns:
            synthetic[column] = selected[column].round().astype(int)

    if "MARS" not in synthetic.columns or "XTOT" not in synthetic.columns:
        return
    joint = synthetic["MARS"] == 2
    synthetic.loc[joint, "XTOT"] = np.maximum(synthetic.loc[joint, "XTOT"], 2)
    synthetic["XTOT"] = synthetic["XTOT"].clip(lower=0, upper=5).astype(int)


def _calibrate_amount_columns(
    synthetic: pd.DataFrame,
    selected: pd.DataFrame,
    row: pd.Series,
    recid: int,
    pop_weight: float,
    target_total_agi: float,
    amount_columns: list[str],
    synthetic_weights: np.ndarray,
    spec: PufAggregateDisaggregationSpec,
) -> None:
    synthetic["E00100"] = _allocate_agi_values(
        donor_agi=selected["E00100"].to_numpy(dtype=float),
        weights=synthetic_weights,
        recid=recid,
        target_total=target_total_agi,
        spec=spec,
    )
    for column in amount_columns:
        if column == "E00100":
            continue
        target_total = pop_weight * _finite_float(row.get(column, 0.0))
        synthetic[column] = _allocate_weighted_values(
            base_values=selected[column].to_numpy(dtype=float),
            weights=synthetic_weights,
            target_total=target_total,
        )
        _assert_weighted_total_matches(
            column=column,
            recid=recid,
            values=synthetic[column].to_numpy(dtype=float),
            weights=synthetic_weights,
            target_total=target_total,
        )


def _disaggregate_bucket(
    *,
    recid: int,
    row: pd.Series,
    regular: pd.DataFrame,
    amount_columns: list[str],
    donor_scores: pd.Series,
    next_recid: int,
    rng: np.random.Generator,
    spec: PufAggregateDisaggregationSpec,
) -> pd.DataFrame:
    pop_weight, target_mean_agi, target_total_agi = _get_bucket_targets(row)
    donor_bucket = _get_donor_bucket(regular, recid, spec)
    total_s006 = int(round(_finite_float(row["S006"])))
    n_synthetic = min(_choose_n_synthetic(pop_weight), max(total_s006, 1))
    synthetic_s006 = _assign_s006_values(
        total_s006,
        n_synthetic,
        rng,
    )
    synthetic_weights = synthetic_s006.astype(float) / 100.0

    selected = _sample_bucket_donors(
        donor_bucket=donor_bucket,
        donor_scores=donor_scores,
        target_mean_agi=target_mean_agi,
        n_synthetic=n_synthetic,
        rng=rng,
    )
    selected = _coerce_amount_columns(selected, amount_columns)

    synthetic = selected.copy()
    synthetic["RECID"] = np.arange(next_recid, next_recid + n_synthetic, dtype=int)
    synthetic["S006"] = synthetic_s006
    _apply_structural_templates(synthetic, selected)
    _calibrate_amount_columns(
        synthetic=synthetic,
        selected=selected,
        row=row,
        recid=recid,
        pop_weight=pop_weight,
        target_total_agi=target_total_agi,
        amount_columns=amount_columns,
        synthetic_weights=synthetic_weights,
        spec=spec,
    )
    return synthetic
