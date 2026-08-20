"""Recover TY2015 AGI-band values aligned to the processed PUF artifact.

The processed 2024 PUF is rooted in the 2015 IRS PUF but does not export raw
``E00100``. Most processed tax units retain their IRS ``RECID`` and therefore
join directly to raw E00100. The remaining IDs replace the four IRS disclosure
aggregate rows. This module reproduces the archived, seeded donor allocation
for the three bounded aggregate buckets; the Forbes-backed $100 million-plus
tail receives that bucket's lower bound because every such record maps to the
same final SOI Table 2.1 band ($10 million or more).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

PUF_SOURCE_YEAR = 2015
PUF_AGGREGATE_RECIDS = (999_996, 999_997, 999_998, 999_999)
PUF_SYNTHETIC_RECID_START = 1_000_000
PUF_AGGREGATE_DISAGGREGATION_SEED = 42
PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS = (
    "RECID",
    "MARS",
    "S006",
    "E00100",
    "E00200",
    "P23250",
    "P22250",
    "E00650",
    "E00300",
    "E26270",
    "E00900",
    "E02100",
    "E00400",
    "E00600",
)

_SCREENED_FIELDS = PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS[4:]
_SELECTION_POWER = 24
_MAX_AGI_DOMINANCE = 0.20
_AGI_CAP_100M_PLUS = 1_250_000_000.0
_NUMERIC_TOLERANCE = 1e-9
_AGGREGATE_BUCKET_BOUNDS = {
    999_996: (None, 0.0),
    999_997: (0.0, 10_000_000.0),
    999_998: (10_000_000.0, 100_000_000.0),
    999_999: (100_000_000.0, None),
}


def _source_frame(
    source: str | Path | BinaryIO | pd.DataFrame | Mapping[str, Sequence[Any]],
) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        frame = pd.read_csv(
            source_path,
            usecols=lambda name: name in PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS,
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        source.seek(0)
        frame = pd.read_csv(
            source,
            usecols=lambda name: name in PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS,
        )
    else:
        frame = pd.DataFrame(source).copy()
    missing = [
        column for column in PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS if column not in frame
    ]
    if missing:
        raise ValueError(f"TY2015 PUF source is missing column(s): {missing}.")
    frame = frame.loc[:, PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS].copy()
    for column in PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS[:4])].isna().any().any():
        raise ValueError("TY2015 PUF RECID, MARS, S006, and E00100 must be numeric.")
    frame[list(_SCREENED_FIELDS)] = frame[list(_SCREENED_FIELDS)].fillna(0.0)
    return frame


def load_source_year_puf_frame(
    source: str | Path | BinaryIO,
) -> pd.DataFrame:
    """Parse the restricted TY2015 columns without running seeded transforms."""

    return _source_frame(source)


def _integral_ids(values: Sequence[Any], *, label: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise ValueError(f"{label} must be a finite one-dimensional array.")
    integral = numeric.astype(np.int64)
    if not np.array_equal(numeric, integral.astype(np.float64)):
        raise ValueError(f"{label} must contain integral values.")
    if len(np.unique(integral)) != len(integral):
        raise ValueError(f"{label} must be unique.")
    return integral


def _choose_n_synthetic(population_weight: float) -> int:
    return int(min(40, max(20, round(population_weight / 10))))


def _assign_weights(
    population_weight: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_weight = int(round(population_weight))
    base = max(total_weight // count, 3)
    weights = np.full(count, base, dtype=int)
    remainder = total_weight - int(weights.sum())
    if remainder > 0:
        weights[rng.choice(count, size=remainder, replace=False)] += 1
    elif remainder < 0:
        reducible = np.where(weights > 1)[0]
        reduce_count = min(-remainder, len(reducible))
        weights[rng.choice(reducible, size=reduce_count, replace=False)] -= 1
    gap = total_weight - int(weights.sum())
    if gap:
        weights[0] += gap
    return weights.astype(np.float64)


def _bucket_mask(frame: pd.DataFrame, recid: int) -> pd.Series:
    agi = frame["E00100"]
    if recid == 999_996:
        return agi < 0
    if recid == 999_997:
        return (agi >= 0) & (agi < 10_000_000)
    if recid == 999_998:
        return (agi >= 10_000_000) & (agi < 100_000_000)
    if recid == 999_999:
        return agi >= 100_000_000
    raise ValueError(f"Unknown aggregate RECID {recid}.")


def _eligibility_scores(frame: pd.DataFrame) -> pd.Series:
    maximum = np.zeros(len(frame), dtype=np.float64)
    for field in _SCREENED_FIELDS:
        values = frame[field].fillna(0.0).to_numpy(dtype=np.float64)
        scores = np.zeros(len(frame), dtype=np.float64)

        positive = values > 0
        reference_positive = np.sort(values[positive])
        if positive.any():
            scores[positive] = np.searchsorted(
                reference_positive, values[positive], side="right"
            ) / len(reference_positive)

        negative = values < 0
        reference_negative = np.sort(-values[negative])
        if negative.any():
            scores[negative] = np.maximum(
                scores[negative],
                np.searchsorted(
                    reference_negative,
                    -values[negative],
                    side="right",
                )
                / len(reference_negative),
            )
        maximum = np.maximum(maximum, scores)
    return pd.Series(maximum, index=frame.index, dtype=np.float64)


def _sample_bucket_donors(
    regular: pd.DataFrame,
    scores: pd.Series,
    *,
    recid: int,
    target_mean_agi: float,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    donor_bucket = regular.loc[_bucket_mask(regular, recid)]
    if donor_bucket.empty:
        donor_bucket = regular
    bucket_scores = scores.loc[donor_bucket.index].to_numpy(dtype=np.float64)
    score_mass = np.clip(bucket_scores, 1e-6, None) ** _SELECTION_POWER
    donor_abs_agi = np.abs(donor_bucket["E00100"].to_numpy(dtype=np.float64))
    target_abs_agi = max(abs(float(target_mean_agi)), 1.0)
    distance = np.abs(np.log1p(donor_abs_agi) - np.log1p(target_abs_agi))
    probabilities = score_mass * np.sqrt(1.0 / (1.0 + distance))
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        probabilities = np.ones(len(donor_bucket), dtype=np.float64)
    probabilities /= probabilities.sum()
    selected_index = rng.choice(
        donor_bucket.index.to_numpy(),
        size=count,
        replace=len(donor_bucket) < count,
        p=probabilities,
    )
    return donor_bucket.loc[selected_index].reset_index(drop=True)


def _project_weighted_sum_to_bounds(
    values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    projected = np.clip(values.astype(np.float64), lower, upper)
    for _ in range(50):
        residual = float(target_total - np.dot(projected, weights))
        if abs(residual) <= 1e-6:
            return projected
        slack = upper - projected if residual > 0 else projected - lower
        free = slack > _NUMERIC_TOLERANCE
        if not free.any():
            break
        basis = np.abs(projected[free])
        if basis.sum() <= _NUMERIC_TOLERANCE:
            basis = np.ones(free.sum(), dtype=np.float64)
        denominator = float(np.dot(weights[free], basis))
        if denominator <= _NUMERIC_TOLERANCE:
            basis = np.ones(free.sum(), dtype=np.float64)
            denominator = float(weights[free].sum())
        delta = residual * basis / denominator
        if residual > 0:
            delta = np.minimum(delta, slack[free])
        else:
            delta = -np.minimum(-delta, slack[free])
        projected[free] += delta
        projected = np.clip(projected, lower, upper)

    residual = float(target_total - np.dot(projected, weights))
    if abs(residual) > 1e-6:
        slack = upper - projected if residual > 0 else projected - lower
        free = np.where(slack > _NUMERIC_TOLERANCE)[0]
        if len(free):
            best = free[np.argmax(slack[free] * weights[free])]
            projected[best] = np.clip(
                projected[best] + residual / weights[best],
                lower[best],
                upper[best],
            )
    return projected


def _allocate_agi(
    donor_agi: np.ndarray,
    weights: np.ndarray,
    *,
    recid: int,
    target_total: float,
) -> np.ndarray:
    dominance_cap = _MAX_AGI_DOMINANCE * abs(target_total) / weights
    lower_bound, upper_bound = _AGGREGATE_BUCKET_BOUNDS[recid]
    if recid == 999_996:
        lower = -dominance_cap
        upper = np.zeros(len(weights), dtype=np.float64)
    else:
        lower = np.full(len(weights), float(lower_bound), dtype=np.float64)
        numeric_upper = (
            _AGI_CAP_100M_PLUS if upper_bound is None else float(upper_bound)
        )
        upper = np.minimum(
            np.full(len(weights), numeric_upper, dtype=np.float64),
            dominance_cap,
        )

    base = np.abs(np.asarray(donor_agi, dtype=np.float64))
    active = base > _NUMERIC_TOLERANCE
    if not active.any():
        active = np.ones(len(base), dtype=bool)
    allocated = np.zeros(len(base), dtype=np.float64)
    magnitudes = base[active]
    if magnitudes.sum() <= _NUMERIC_TOLERANCE:
        magnitudes = np.ones(active.sum(), dtype=np.float64)
    denominator = float(np.dot(weights[active], magnitudes))
    if denominator <= _NUMERIC_TOLERANCE:
        magnitudes = np.ones(active.sum(), dtype=np.float64)
        denominator = float(weights[active].sum())
    allocated[active] = (
        np.sign(target_total) * magnitudes * abs(target_total) / denominator
    )
    return _project_weighted_sum_to_bounds(
        allocated,
        weights,
        target_total,
        lower,
        upper,
    )


def source_year_puf_adjusted_gross_income(
    source: str | Path | BinaryIO | pd.DataFrame | Mapping[str, Sequence[Any]],
    *,
    processed_tax_unit_ids: Sequence[Any],
    processed_tax_unit_weights: Sequence[Any],
) -> np.ndarray:
    """Return TY2015 AGI banding values in processed-PUF tax-unit order.

    Regular tax units carry literal IRS PUF E00100. The first three synthetic
    disclosure buckets reproduce the archived seeded donor allocation. The
    open Forbes-backed tail receives $100 million, its source bucket floor;
    this is an AGI-band anchor rather than a fabricated point estimate, and all
    such rows belong to Table 2.1's final "$10 million or more" band.
    """

    frame = _source_frame(source)
    source_ids = _integral_ids(frame["RECID"], label="TY2015 PUF RECID")
    frame["RECID"] = source_ids
    if set(frame.loc[frame["MARS"] == 0, "RECID"]) != set(PUF_AGGREGATE_RECIDS):
        raise ValueError("TY2015 PUF must contain exactly the four aggregate RECIDs.")

    processed_ids = _integral_ids(
        processed_tax_unit_ids,
        label="Processed PUF tax_unit_id",
    )
    processed_weights = np.asarray(processed_tax_unit_weights, dtype=np.float64)
    if (
        processed_weights.ndim != 1
        or len(processed_weights) != len(processed_ids)
        or not np.isfinite(processed_weights).all()
        or (processed_weights <= 0).any()
    ):
        raise ValueError(
            "Processed PUF tax-unit weights must be finite, positive, and "
            "aligned one-for-one with tax_unit_id."
        )

    aggregate_mask = frame["RECID"].isin(PUF_AGGREGATE_RECIDS)
    regular = frame.loc[~aggregate_mask].copy()
    regular_ids = regular["RECID"].to_numpy(dtype=np.int64)
    if len(processed_ids) <= len(regular_ids) or not np.array_equal(
        processed_ids[: len(regular_ids)],
        regular_ids,
    ):
        raise ValueError(
            "Processed PUF regular RECID order does not match the TY2015 source."
        )

    synthetic_ids = processed_ids[len(regular_ids) :]
    expected_synthetic_ids = np.arange(
        PUF_SYNTHETIC_RECID_START,
        PUF_SYNTHETIC_RECID_START + len(synthetic_ids),
        dtype=np.int64,
    )
    if not np.array_equal(synthetic_ids, expected_synthetic_ids):
        raise ValueError(
            "Processed PUF synthetic tax_unit_id values are not the archived "
            "contiguous aggregate-record replacement IDs."
        )

    source_regular_weights = regular["S006"].to_numpy(dtype=np.float64) / 100.0
    if (source_regular_weights <= 0).any():
        raise ValueError("TY2015 regular PUF S006 weights must be positive.")
    weight_ratios = processed_weights[: len(regular_ids)] / source_regular_weights
    population_scale = float(np.median(weight_ratios))
    if not np.allclose(
        weight_ratios,
        population_scale,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Processed PUF regular weights do not share one source uprating factor."
        )

    aggregate = frame.loc[aggregate_mask].set_index("RECID")
    aggregate_population = {
        recid: float(aggregate.loc[recid, "S006"]) / 100.0 * population_scale
        for recid in PUF_AGGREGATE_RECIDS
    }
    bounded_counts = {
        recid: _choose_n_synthetic(aggregate_population[recid])
        for recid in PUF_AGGREGATE_RECIDS[:3]
    }
    top_count = len(synthetic_ids) - sum(bounded_counts.values())
    if top_count <= 0:
        raise ValueError(
            "Processed PUF does not contain the archived open-tail replacement rows."
        )
    counts = {
        **bounded_counts,
        PUF_AGGREGATE_RECIDS[-1]: top_count,
    }

    output = np.empty(len(processed_ids), dtype=np.float64)
    output[: len(regular_ids)] = regular["E00100"].to_numpy(dtype=np.float64)
    if not np.isfinite(output[: len(regular_ids)]).all():
        raise ValueError("TY2015 regular PUF E00100 must be finite.")

    scores = _eligibility_scores(regular)
    rng = np.random.default_rng(PUF_AGGREGATE_DISAGGREGATION_SEED)
    cursor = len(regular_ids)
    for recid in PUF_AGGREGATE_RECIDS[:3]:
        count = counts[recid]
        assigned_weights = _assign_weights(
            aggregate_population[recid],
            count,
            rng,
        )
        processed_bucket_weights = processed_weights[cursor : cursor + count]
        if not np.array_equal(assigned_weights, processed_bucket_weights):
            raise ValueError(
                f"Processed PUF synthetic weights for aggregate RECID {recid} "
                "do not match the archived seeded assignment."
            )
        target_mean_agi = float(aggregate.loc[recid, "E00100"])
        selected = _sample_bucket_donors(
            regular,
            scores,
            recid=recid,
            target_mean_agi=target_mean_agi,
            count=count,
            rng=rng,
        )
        output[cursor : cursor + count] = _allocate_agi(
            selected["E00100"].to_numpy(dtype=np.float64),
            assigned_weights,
            recid=recid,
            target_total=aggregate_population[recid] * target_mean_agi,
        )
        cursor += count

    top_weights = processed_weights[cursor:]
    expected_top_weight = int(round(aggregate_population[999_999]))
    if not np.isclose(top_weights.sum(), expected_top_weight, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Processed PUF open-tail weights do not preserve the archived "
            "aggregate-row population."
        )
    output[cursor:] = _AGGREGATE_BUCKET_BOUNDS[999_999][0]
    if not np.isfinite(output).all():
        raise ValueError("Recovered TY2015 PUF AGI banding values must be finite.")
    return output


__all__ = [
    "PUF_AGGREGATE_DISAGGREGATION_SEED",
    "PUF_AGGREGATE_RECIDS",
    "PUF_SOURCE_YEAR",
    "PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS",
    "PUF_SYNTHETIC_RECID_START",
    "load_source_year_puf_frame",
    "source_year_puf_adjusted_gross_income",
]
