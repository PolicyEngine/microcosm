"""Build and validate the SCF shape for passive pass-through income.

The public 2022 Survey of Consumer Finances (SCF) does not identify passive
income by tax entity.  It does, however, separately identify households that
actively manage a business and households that hold a business without an
active management role.  This module turns those observations into the small,
deterministic evidence surface used by the NIIT passive pass-through stage:

* the probability of holding any non-actively-managed business, conditional
  on having an actively managed business and an ``X5714`` income band; and
* the conditional share of business value held outside active management,
  ``NONACTBUS / (ACTBUS + NONACTBUS)``.

All five SCF implicates are pooled and ``X42001`` is divided by five.  A band
estimate is retained only when its implicate-adjusted unweighted sample is at
least 30; otherwise it collapses to the all-band sample.  This is evidence
construction only.  Assignment and administrative-level calibration live in
the sibling passive pass-through runtime.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCF_PASSIVE_IMPLICATE_COUNT = 5
SCF_PASSIVE_MINIMUM_EFFECTIVE_N = 30.0
SCF_PASSIVE_QUANTILE_PROBABILITIES = (0.05, 0.25, 0.50, 0.75, 0.95)
SCF_PASSIVE_INCOME_BANDS = (
    "nonpositive",
    "0_to_25k",
    "25k_to_100k",
    "100k_to_250k",
    "250k_to_1m",
    "over_1m",
)

SCF_PASSIVE_REQUIRED_COLUMNS = (
    "y1",
    "x42001",
    "x3103",
    "x3104",
    "x3105",
    "x3401",
    "x5714",
    "x3129",
    "x3124",
    "x3126",
    "x3127",
    "x3121",
    "x3122",
    "x3229",
    "x3224",
    "x3226",
    "x3227",
    "x3221",
    "x3222",
    "x3335",
    "x3408",
    "x3412",
    "x3416",
    "x3420",
    "x3452",
    "x3428",
    "x507",
    "x513",
    "x526",
    "x805",
    "x905",
    "x1005",
    "x1103",
    "x1108",
    "x1114",
    "x1119",
    "x1125",
    "x1130",
    "x1136",
)

FORM_8960_TAX_YEAR = 2023
FORM_8960_LINE_4A_AMOUNT = 1_185_607_258_000.0
FORM_8960_LINE_4A_RETURNS = 4_988_033
FORM_8960_LINE_4B_AMOUNT = -1_076_350_273_000.0
FORM_8960_LINE_4B_RETURNS = 4_038_235
FORM_8960_LINE_4C_AMOUNT = 109_256_984_000.0
FORM_8960_LINE_4C_RETURNS = 2_260_296
FORM_8960_LINE_4C_TO_4A_RATIO = (
    FORM_8960_LINE_4C_AMOUNT / FORM_8960_LINE_4A_AMOUNT
)

_QUANTILE_NAMES = ("q05", "q25", "q50", "q75", "q95")
_RESOURCE_NAME = "qbi_passive_passthrough_v1.json"


def _normalized_scf_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={str(column): str(column).lower() for column in frame.columns}
    )
    duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(
            "SCF frame has case-insensitive duplicate columns "
            f"{sorted(set(duplicates))}."
        )
    missing = sorted(set(SCF_PASSIVE_REQUIRED_COLUMNS) - set(normalized.columns))
    if missing:
        raise ValueError(f"SCF frame is missing required columns {missing}.")

    result = normalized.loc[:, SCF_PASSIVE_REQUIRED_COLUMNS].copy()
    for column in SCF_PASSIVE_REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"SCF column {column!r} contains nonnumeric or nonfinite values."
            )
    return result


def _validate_implicates(source: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw_y1 = source["y1"].to_numpy(dtype=np.float64)
    y1 = raw_y1.astype(np.int64)
    if not np.array_equal(raw_y1, y1.astype(np.float64)):
        raise ValueError("SCF Y1 must contain integer identifiers.")
    if pd.Series(y1).duplicated().any():
        raise ValueError("SCF Y1 must be unique across household implicates.")

    household_id = y1 // 10
    implicate = y1 % 10
    expected = set(range(1, SCF_PASSIVE_IMPLICATE_COUNT + 1))
    malformed = [
        int(identifier)
        for identifier, group in pd.DataFrame(
            {"household_id": household_id, "implicate": implicate}
        ).groupby("household_id")["implicate"]
        if set(group.tolist()) != expected
    ]
    if malformed:
        raise ValueError(
            "SCF Y1 must encode exactly implicates 1-5 for every household; "
            f"malformed household ids include {malformed[:5]}."
        )
    return household_id, implicate


def _positive(source: pd.DataFrame, column: str) -> np.ndarray:
    return np.maximum(source[column].to_numpy(dtype=np.float64), 0.0)


def _farm_business_value(source: pd.DataFrame) -> np.ndarray:
    """Reproduce the SCFP ``FARMBUS`` balance-sheet construction."""

    x507 = source["x507"].to_numpy(dtype=np.float64)
    business_fraction = np.where(x507 > 0.0, np.minimum(x507, 9_000.0) / 10_000.0, 0.0)
    farm = business_fraction * (
        source["x513"].to_numpy(dtype=np.float64)
        + source["x526"].to_numpy(dtype=np.float64)
        - source["x805"].to_numpy(dtype=np.float64)
        - source["x905"].to_numpy(dtype=np.float64)
        - source["x1005"].to_numpy(dtype=np.float64)
    )

    first = source["x1108"].to_numpy(dtype=np.float64)
    second = source["x1119"].to_numpy(dtype=np.float64)
    third = source["x1130"].to_numpy(dtype=np.float64)
    first_farm = source["x1103"].to_numpy(dtype=np.float64) == 1.0
    second_farm = source["x1114"].to_numpy(dtype=np.float64) == 1.0
    third_farm = source["x1125"].to_numpy(dtype=np.float64) == 1.0
    farm -= business_fraction * np.where(first_farm, first, 0.0)
    farm -= business_fraction * np.where(second_farm, second, 0.0)
    farm -= business_fraction * np.where(third_farm, third, 0.0)

    # The SCFP macro mutates each farm-secured line-of-credit balance after
    # subtracting its business share.  Its subsequent X1136 allocation uses
    # those reduced copies, not the original balances.
    first_reduced = np.where(first_farm, first * (1.0 - business_fraction), first)
    second_reduced = np.where(
        second_farm, second * (1.0 - business_fraction), second
    )
    third_reduced = np.where(third_farm, third * (1.0 - business_fraction), third)
    other_mortgage = source["x1136"].to_numpy(dtype=np.float64)
    mortgage_total = first_reduced + second_reduced + third_reduced
    farm_mortgage = (
        first_reduced * first_farm.astype(np.float64)
        + second_reduced * second_farm.astype(np.float64)
        + third_reduced * third_farm.astype(np.float64)
    )
    shared_fraction = np.divide(
        farm_mortgage,
        mortgage_total,
        out=np.zeros_like(farm_mortgage),
        where=mortgage_total > 0.0,
    )
    farm -= np.where(
        (other_mortgage > 0.0) & (mortgage_total > 0.0),
        other_mortgage * business_fraction * shared_fraction,
        0.0,
    )
    return farm


def _active_business_value(source: pd.DataFrame) -> np.ndarray:
    first = (
        _positive(source, "x3129")
        + _positive(source, "x3124")
        - _positive(source, "x3126")
        * (source["x3127"].to_numpy(dtype=np.float64) == 5.0)
        + _positive(source, "x3121")
        * np.isin(source["x3122"].to_numpy(dtype=np.float64), (1.0, 6.0))
    )
    second = (
        _positive(source, "x3229")
        + _positive(source, "x3224")
        - _positive(source, "x3226")
        * (source["x3227"].to_numpy(dtype=np.float64) == 5.0)
        + _positive(source, "x3221")
        * np.isin(source["x3222"].to_numpy(dtype=np.float64), (1.0, 6.0))
    )
    return first + second + _positive(source, "x3335") + _farm_business_value(source)


def _nonactive_business_value(source: pd.DataFrame) -> np.ndarray:
    columns = ("x3408", "x3412", "x3416", "x3420", "x3452", "x3428")
    return sum((_positive(source, column) for column in columns), np.zeros(len(source)))


def _income_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=(-np.inf, 0.0, 25_000.0, 100_000.0, 250_000.0, 1_000_000.0, np.inf),
        labels=SCF_PASSIVE_INCOME_BANDS,
        right=True,
        include_lowest=True,
    ).astype("object")


def build_scf_passive_passthrough_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Return pooled active-business household-implicate records from SCF."""

    source = _normalized_scf_frame(frame)
    household_id, implicate = _validate_implicates(source)
    raw_weight = source["x42001"].to_numpy(dtype=np.float64)
    if (raw_weight <= 0.0).any():
        raise ValueError("SCF X42001 must be positive for every implicate.")

    active = (
        source["x3103"].eq(1)
        & source["x3104"].eq(1)
        & source["x3105"].ge(1)
    ).to_numpy()
    if not active.any():
        raise ValueError("SCF frame contains no actively managed business households.")

    actbus = _active_business_value(source)
    nonactbus = _nonactive_business_value(source)
    denominator = actbus + nonactbus
    holder = source["x3401"].eq(1).to_numpy()
    eligible_share = holder & (denominator > 0.0)
    share = np.divide(
        nonactbus,
        denominator,
        out=np.full(len(source), np.nan, dtype=np.float64),
        where=eligible_share,
    )
    invalid_share = eligible_share & ((share < 0.0) | (share > 1.0))
    if invalid_share.any():
        raise ValueError(
            "SCF NONACTBUS / (ACTBUS + NONACTBUS) falls outside [0, 1] "
            f"for {int(invalid_share.sum())} eligible record(s)."
        )

    records = pd.DataFrame(
        {
            "household_id": household_id,
            "implicate": implicate,
            "weight": raw_weight / SCF_PASSIVE_IMPLICATE_COUNT,
            "schedule_e_income": source["x5714"].to_numpy(dtype=np.float64),
            "income_band": _income_band(source["x5714"]),
            "holds_nonactive_business": holder,
            "actbus": actbus,
            "nonactbus": nonactbus,
            "positive_share_denominator": denominator > 0.0,
            "nonactive_business_value_share": share,
        }
    )
    return records.loc[active].reset_index(drop=True)


def weighted_inverse_cdf(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] = SCF_PASSIVE_QUANTILE_PROBABILITIES,
) -> dict[str, float]:
    """Return weighted inverse-CDF quantiles with stable tie handling."""

    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if value_array.ndim != 1 or value_array.shape != weight_array.shape:
        raise ValueError("Weighted values and weights must be equal 1-D arrays.")
    if value_array.size == 0:
        raise ValueError("Weighted quantiles require at least one observation.")
    if (
        not np.isfinite(value_array).all()
        or not np.isfinite(weight_array).all()
        or (weight_array <= 0.0).any()
    ):
        raise ValueError("Weighted quantiles require finite values and positive weights.")
    if (
        not np.isfinite(probability_array).all()
        or (probability_array < 0.0).any()
        or (probability_array > 1.0).any()
    ):
        raise ValueError("Weighted quantile probabilities must lie in [0, 1].")

    order = np.argsort(value_array, kind="stable")
    cumulative = np.cumsum(weight_array[order])
    positions = np.searchsorted(
        cumulative, probability_array * cumulative[-1], side="left"
    )
    positions = np.minimum(positions, len(order) - 1)
    return {
        name: float(value_array[order[position]])
        for name, position in zip(_QUANTILE_NAMES, positions, strict=True)
    }


def _counts(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "pooled_record_count": int(len(frame)),
        "implicate_adjusted_unweighted_n": float(
            len(frame) / SCF_PASSIVE_IMPLICATE_COUNT
        ),
        "weighted_households": float(frame["weight"].sum()),
    }


def _weighted_mean(frame: pd.DataFrame) -> float:
    return float(
        np.average(
            frame["nonactive_business_value_share"].to_numpy(dtype=np.float64),
            weights=frame["weight"].to_numpy(dtype=np.float64),
        )
    )


def _quantiles(frame: pd.DataFrame) -> dict[str, float]:
    return weighted_inverse_cdf(
        frame["nonactive_business_value_share"].to_numpy(dtype=np.float64),
        frame["weight"].to_numpy(dtype=np.float64),
    )


def _selected_sample(
    requested: pd.DataFrame,
    pooled: pd.DataFrame,
) -> tuple[str, str, pd.DataFrame]:
    effective_n = len(requested) / SCF_PASSIVE_IMPLICATE_COUNT
    if effective_n >= SCF_PASSIVE_MINIMUM_EFFECTIVE_N:
        return "exact", str(requested["income_band"].iloc[0]), requested
    if pooled.empty:
        raise ValueError("SCF all-band fallback sample is empty.")
    return "all_income_bands", "all", pooled


def _requested_counts(
    active: pd.DataFrame,
    holders: pd.DataFrame,
    eligible_holders: pd.DataFrame,
) -> dict[str, float | int]:
    counts: dict[str, float | int] = {}
    for prefix, sample in (
        ("active", active),
        ("nonactive_holder", holders),
        ("positive_denominator_holder", eligible_holders),
    ):
        sample_counts = _counts(sample)
        for name, value in sample_counts.items():
            counts[f"{prefix}_{name}"] = value
    return counts


def _source_description() -> dict[str, object]:
    return {
        "survey": "Federal Reserve Survey of Consumer Finances",
        "vintage": 2022,
        "public_extract": "full public data, pooled multiple-imputation file p22i6.dta",
        "weight": "X42001 divided by five implicates",
        "record_selection": (
            "X3103 == 1 and X3104 == 1 and X3105 >= 1; one record per "
            "household implicate. A non-active holding is X3401 == 1, even "
            "when the reported NONACTBUS value is zero."
        ),
        "variables": {
            "schedule_e_income_band": "X5714",
            "active_management_screeners": ["X3103", "X3104", "X3105"],
            "nonactive_holding_screener": "X3401",
            "active_business_value": (
                "SCFP ACTBUS: active-business net values and family-business "
                "loan adjustments for X3121-X3335, plus FARMBUS"
            ),
            "nonactive_business_value": (
                "SCFP NONACTBUS: sum of nonnegative X3408, X3412, X3416, "
                "X3420, X3452, and X3428"
            ),
        },
    }


def _methodology() -> dict[str, object]:
    return {
        "estimands": {
            "holding_prevalence": (
                "P(X3401 == 1 | actively managed business, X5714 income band)"
            ),
            "conditional_share": (
                "NONACTBUS / (ACTBUS + NONACTBUS), conditional on X3401 == 1 "
                "and a positive denominator"
            ),
        },
        "income_bands": [
            {"id": "nonpositive", "definition": "X5714 <= 0"},
            {"id": "0_to_25k", "definition": "0 < X5714 <= 25,000"},
            {"id": "25k_to_100k", "definition": "25,000 < X5714 <= 100,000"},
            {
                "id": "100k_to_250k",
                "definition": "100,000 < X5714 <= 250,000",
            },
            {
                "id": "250k_to_1m",
                "definition": "250,000 < X5714 <= 1,000,000",
            },
            {"id": "over_1m", "definition": "X5714 > 1,000,000"},
        ],
        "implicate_count": SCF_PASSIVE_IMPLICATE_COUNT,
        "minimum_effective_n": SCF_PASSIVE_MINIMUM_EFFECTIVE_N,
        "fallback_order": ["income_band", "all_income_bands"],
        "quantile_probabilities": list(SCF_PASSIVE_QUANTILE_PROBABILITIES),
        "quantile_method": "weighted inverse CDF",
    }


def _external_anchor() -> dict[str, object]:
    return {
        "source": "IRS Publication 4801, Form 8960 statistics, tax year 2023",
        "units": "dollars",
        "form_8960": {
            "line_4a": {
                "description": (
                    "gross rental, royalty, partnership, S corporation, trust, "
                    "and trade-or-business flow-through"
                ),
                "returns_count": FORM_8960_LINE_4A_RETURNS,
                "amount": FORM_8960_LINE_4A_AMOUNT,
            },
            "line_4b": {
                "description": "non-section-1411 trade-or-business amount removed",
                "returns_count": FORM_8960_LINE_4B_RETURNS,
                "amount": FORM_8960_LINE_4B_AMOUNT,
            },
            "line_4c": {
                "description": "line 4a plus line 4b; amount remaining in the NIIT base",
                "returns_count": FORM_8960_LINE_4C_RETURNS,
                "amount": FORM_8960_LINE_4C_AMOUNT,
            },
        },
        "line_4c_to_4a_survival_ratio": FORM_8960_LINE_4C_TO_4A_RATIO,
        "passive_passthrough_bounds": {
            "lower": {
                "amount": 0.0,
                "assumption": "all Form 8960 line 4c is rental or royalty income",
            },
            "upper": {
                "amount": FORM_8960_LINE_4C_AMOUNT,
                "assumption": "all Form 8960 line 4c is passive pass-through income",
            },
            "decomposition_status": (
                "open question: no administrative source on disk decomposes line "
                "4c between rental or royalty income and passive partnership or "
                "S-corporation income"
            ),
        },
        "caveats": [
            (
                "Line 4c includes rental and royalty net investment income as well "
                "as passive partnership and S-corporation income; it is not a "
                "passive pass-through point estimate."
            ),
            (
                "The engine counts rental_income in NIIT in full. Schedule E rental "
                "income is frequently passive, so using line 4c without a rental "
                "decomposition would overlap an existing NIIT leg. Rental handling "
                "is unchanged here."
            ),
            (
                "No entity-side SOI passive split exists: partnership tables "
                "23pa01, 23pa04, 23pa06, 23pa10, and 23pa23 were checked, and "
                "IRS Table 1.4 likewise has no passive split. Form 8960 line 4 "
                "is the only administrative level available for this provisional "
                "evidence surface."
            ),
        ],
    }


def build_qbi_passive_passthrough_resource(
    frame: pd.DataFrame,
    *,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic version-1 passive pass-through evidence resource."""

    records = build_scf_passive_passthrough_records(frame)
    pooled_holders = records.loc[
        records["holds_nonactive_business"]
        & records["positive_share_denominator"]
    ]
    if pooled_holders.empty:
        raise ValueError("SCF records contain no eligible non-active business holders.")

    cells: list[dict[str, object]] = []
    for income_band in SCF_PASSIVE_INCOME_BANDS:
        active = records.loc[records["income_band"].eq(income_band)]
        if active.empty:
            raise ValueError(f"SCF income band {income_band!r} is empty.")
        holders = active.loc[active["holds_nonactive_business"]]
        eligible_holders = holders.loc[holders["positive_share_denominator"]]
        if eligible_holders.empty:
            raise ValueError(
                f"SCF income band {income_band!r} has no positive-denominator holders."
            )

        presence_level, presence_source_band, presence_sample = _selected_sample(
            active, records
        )
        share_level, share_source_band, share_sample = _selected_sample(
            eligible_holders, pooled_holders
        )
        requested_weight = float(active["weight"].sum())
        selected_presence_weight = float(presence_sample["weight"].sum())
        cells.append(
            {
                "income_band": income_band,
                "requested_counts": _requested_counts(
                    active, holders, eligible_holders
                ),
                "holding_prevalence": {
                    "estimate": float(
                        presence_sample.loc[
                            presence_sample["holds_nonactive_business"], "weight"
                        ].sum()
                        / selected_presence_weight
                    ),
                    "requested_estimate": float(
                        holders["weight"].sum() / requested_weight
                    ),
                    "estimate_level": presence_level,
                    "source_income_band": presence_source_band,
                    "source_counts": _counts(presence_sample),
                },
                "conditional_share": {
                    "conditional_on": (
                        "X3401 == 1 and ACTBUS + NONACTBUS > 0"
                    ),
                    "requested_mean": _weighted_mean(eligible_holders),
                    "requested_quantiles": _quantiles(eligible_holders),
                    "selected_mean": _weighted_mean(share_sample),
                    "selected_quantiles": _quantiles(share_sample),
                    "estimate_level": share_level,
                    "source_income_band": share_source_band,
                    "source_counts": _counts(share_sample),
                },
            }
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "resource": "us_qbi_passive_passthrough_evidence",
        "survey_year": 2022,
        "provisional": True,
        "source": _source_description(),
        "methodology": _methodology(),
        "cells": cells,
        "external_anchor": _external_anchor(),
        "notes": [
            (
                "SCF X3119 identifies the legal form of the largest actively "
                "managed business, not the non-actively-managed holding. There is "
                "no defensible entity-form link, so these estimates are band-only."
            ),
            (
                "The SCF value share is a survey shape, not an income-share level. "
                "The separate assumptions build supplies the provisional "
                "administrative calibration and persists its solved shift."
            ),
            (
                "A household with X3401 == 1 remains a holder when NONACTBUS is "
                "zero; this preserves the requested holding concept."
            ),
        ],
        "provenance": dict(provenance),
    }
    validate_qbi_passive_passthrough_resource(payload)
    return payload


def _require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object.")
    return value


def _require_list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list.")
    return value


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{location} must be finite.")
    return result


def _probability(value: object, location: str) -> float:
    result = _finite_number(value, location)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must lie in [0, 1].")
    return result


def _validate_quantiles(value: object, location: str) -> None:
    quantiles = _require_mapping(value, location)
    if tuple(quantiles) != _QUANTILE_NAMES:
        raise ValueError(f"{location} must have ordered keys {_QUANTILE_NAMES}.")
    values = [_probability(quantiles[name], f"{location}.{name}") for name in quantiles]
    if values != sorted(values):
        raise ValueError(f"{location} must be nondecreasing.")


def _validate_counts(value: object, location: str) -> Mapping[str, Any]:
    counts = _require_mapping(value, location)
    for name in (
        "pooled_record_count",
        "implicate_adjusted_unweighted_n",
        "weighted_households",
    ):
        number = _finite_number(counts.get(name), f"{location}.{name}")
        if number < 0.0:
            raise ValueError(f"{location}.{name} must be nonnegative.")
    pooled = _finite_number(counts["pooled_record_count"], f"{location}.pooled_record_count")
    effective = _finite_number(
        counts["implicate_adjusted_unweighted_n"],
        f"{location}.implicate_adjusted_unweighted_n",
    )
    if not np.isclose(effective, pooled / SCF_PASSIVE_IMPLICATE_COUNT):
        raise ValueError(f"{location} has inconsistent implicate-adjusted n.")
    return counts


def validate_qbi_passive_passthrough_resource(
    payload: Mapping[str, object],
) -> None:
    """Reject malformed or silently de-provisionalized evidence payloads."""

    root = _require_mapping(payload, "resource")
    if root.get("schema_version") != 1:
        raise ValueError("resource.schema_version must equal 1.")
    if root.get("resource") != "us_qbi_passive_passthrough_evidence":
        raise ValueError("resource.resource has the wrong identifier.")
    if root.get("survey_year") != 2022:
        raise ValueError("resource.survey_year must equal 2022.")
    if root.get("provisional") is not True:
        raise ValueError("Passive pass-through evidence must remain provisional.")

    methodology = _require_mapping(root.get("methodology"), "methodology")
    if methodology.get("implicate_count") != SCF_PASSIVE_IMPLICATE_COUNT:
        raise ValueError("methodology.implicate_count must equal 5.")
    if methodology.get("minimum_effective_n") != SCF_PASSIVE_MINIMUM_EFFECTIVE_N:
        raise ValueError("methodology.minimum_effective_n must equal 30.")
    if methodology.get("fallback_order") != ["income_band", "all_income_bands"]:
        raise ValueError("methodology.fallback_order is not the reviewed hierarchy.")
    probabilities = _require_list(
        methodology.get("quantile_probabilities"),
        "methodology.quantile_probabilities",
    )
    if probabilities != list(SCF_PASSIVE_QUANTILE_PROBABILITIES):
        raise ValueError("methodology.quantile_probabilities are not reviewed values.")

    cells = _require_list(root.get("cells"), "cells")
    if [cell.get("income_band") for cell in cells if isinstance(cell, Mapping)] != list(
        SCF_PASSIVE_INCOME_BANDS
    ) or len(cells) != len(SCF_PASSIVE_INCOME_BANDS):
        raise ValueError("cells must contain the six reviewed income bands in order.")
    for index, cell_value in enumerate(cells):
        cell = _require_mapping(cell_value, f"cells[{index}]")
        band = SCF_PASSIVE_INCOME_BANDS[index]
        requested = _require_mapping(
            cell.get("requested_counts"), f"cells[{index}].requested_counts"
        )
        for prefix in (
            "active",
            "nonactive_holder",
            "positive_denominator_holder",
        ):
            _validate_counts(
                {
                    name: requested.get(f"{prefix}_{name}")
                    for name in (
                        "pooled_record_count",
                        "implicate_adjusted_unweighted_n",
                        "weighted_households",
                    )
                },
                f"cells[{index}].requested_counts.{prefix}",
            )

        presence = _require_mapping(
            cell.get("holding_prevalence"), f"cells[{index}].holding_prevalence"
        )
        _probability(presence.get("estimate"), f"cells[{index}].holding_prevalence.estimate")
        _probability(
            presence.get("requested_estimate"),
            f"cells[{index}].holding_prevalence.requested_estimate",
        )
        presence_counts = _validate_counts(
            presence.get("source_counts"),
            f"cells[{index}].holding_prevalence.source_counts",
        )
        active_n = _finite_number(
            requested.get("active_implicate_adjusted_unweighted_n"),
            f"cells[{index}].requested_counts.active_implicate_adjusted_unweighted_n",
        )
        expected_level = (
            "exact"
            if active_n >= SCF_PASSIVE_MINIMUM_EFFECTIVE_N
            else "all_income_bands"
        )
        if presence.get("estimate_level") != expected_level:
            raise ValueError(f"cells[{index}] holding fallback is inconsistent with n.")
        expected_source = band if expected_level == "exact" else "all"
        if presence.get("source_income_band") != expected_source:
            raise ValueError(f"cells[{index}] holding source band is inconsistent.")
        if expected_level == "exact" and not np.isclose(
            _finite_number(
                presence_counts["implicate_adjusted_unweighted_n"],
                f"cells[{index}].holding_prevalence.source_counts.n",
            ),
            active_n,
        ):
            raise ValueError(f"cells[{index}] exact holding source n is inconsistent.")

        share = _require_mapping(
            cell.get("conditional_share"), f"cells[{index}].conditional_share"
        )
        _probability(
            share.get("requested_mean"),
            f"cells[{index}].conditional_share.requested_mean",
        )
        _probability(
            share.get("selected_mean"),
            f"cells[{index}].conditional_share.selected_mean",
        )
        _validate_quantiles(
            share.get("requested_quantiles"),
            f"cells[{index}].conditional_share.requested_quantiles",
        )
        _validate_quantiles(
            share.get("selected_quantiles"),
            f"cells[{index}].conditional_share.selected_quantiles",
        )
        share_counts = _validate_counts(
            share.get("source_counts"),
            f"cells[{index}].conditional_share.source_counts",
        )
        share_n = _finite_number(
            requested.get(
                "positive_denominator_holder_implicate_adjusted_unweighted_n"
            ),
            f"cells[{index}].requested_counts.positive_denominator_holder_n",
        )
        expected_level = (
            "exact"
            if share_n >= SCF_PASSIVE_MINIMUM_EFFECTIVE_N
            else "all_income_bands"
        )
        if share.get("estimate_level") != expected_level:
            raise ValueError(f"cells[{index}] share fallback is inconsistent with n.")
        expected_source = band if expected_level == "exact" else "all"
        if share.get("source_income_band") != expected_source:
            raise ValueError(f"cells[{index}] share source band is inconsistent.")
        if expected_level == "exact" and not np.isclose(
            _finite_number(
                share_counts["implicate_adjusted_unweighted_n"],
                f"cells[{index}].conditional_share.source_counts.n",
            ),
            share_n,
        ):
            raise ValueError(f"cells[{index}] exact share source n is inconsistent.")

    anchor = _require_mapping(root.get("external_anchor"), "external_anchor")
    form = _require_mapping(anchor.get("form_8960"), "external_anchor.form_8960")
    expected_lines = {
        "line_4a": (FORM_8960_LINE_4A_AMOUNT, FORM_8960_LINE_4A_RETURNS),
        "line_4b": (FORM_8960_LINE_4B_AMOUNT, FORM_8960_LINE_4B_RETURNS),
        "line_4c": (FORM_8960_LINE_4C_AMOUNT, FORM_8960_LINE_4C_RETURNS),
    }
    for line, (amount, returns_count) in expected_lines.items():
        row = _require_mapping(form.get(line), f"external_anchor.form_8960.{line}")
        if _finite_number(row.get("amount"), f"{line}.amount") != amount:
            raise ValueError(f"external_anchor.form_8960.{line}.amount is not reviewed.")
        if row.get("returns_count") != returns_count:
            raise ValueError(
                f"external_anchor.form_8960.{line}.returns_count is not reviewed."
            )
    ratio = _finite_number(
        anchor.get("line_4c_to_4a_survival_ratio"),
        "external_anchor.line_4c_to_4a_survival_ratio",
    )
    if not np.isclose(ratio, FORM_8960_LINE_4C_TO_4A_RATIO, rtol=0.0, atol=1e-15):
        raise ValueError("external_anchor survival ratio is inconsistent.")
    bounds = _require_mapping(
        anchor.get("passive_passthrough_bounds"),
        "external_anchor.passive_passthrough_bounds",
    )
    lower = _require_mapping(bounds.get("lower"), "bounds.lower")
    upper = _require_mapping(bounds.get("upper"), "bounds.upper")
    if _finite_number(lower.get("amount"), "bounds.lower.amount") != 0.0:
        raise ValueError("Passive pass-through lower bound must equal zero.")
    if (
        _finite_number(upper.get("amount"), "bounds.upper.amount")
        != FORM_8960_LINE_4C_AMOUNT
    ):
        raise ValueError("Passive pass-through upper bound must equal line 4c.")
    status = bounds.get("decomposition_status")
    if not isinstance(status, str) or "open question" not in status.lower():
        raise ValueError("Bounds must retain the open decomposition question.")

    _require_mapping(root.get("source"), "source")
    _require_list(root.get("notes"), "notes")
    _require_mapping(root.get("provenance"), "provenance")


def default_qbi_passive_passthrough_resource_path() -> Path:
    """Return the packaged evidence resource path."""

    return Path(str(files("microcosm.build.us").joinpath(_RESOURCE_NAME)))


def load_qbi_passive_passthrough_resource(
    path: Path | str | None = None,
) -> dict[str, object]:
    """Load and validate the packaged or explicitly supplied evidence JSON."""

    resource_path = (
        default_qbi_passive_passthrough_resource_path()
        if path is None
        else Path(path)
    )
    payload = json.loads(resource_path.read_text(encoding="utf-8"))
    validate_qbi_passive_passthrough_resource(payload)
    return payload


__all__ = [
    "FORM_8960_LINE_4A_AMOUNT",
    "FORM_8960_LINE_4C_AMOUNT",
    "FORM_8960_LINE_4C_TO_4A_RATIO",
    "SCF_PASSIVE_INCOME_BANDS",
    "SCF_PASSIVE_MINIMUM_EFFECTIVE_N",
    "SCF_PASSIVE_QUANTILE_PROBABILITIES",
    "SCF_PASSIVE_REQUIRED_COLUMNS",
    "build_qbi_passive_passthrough_resource",
    "build_scf_passive_passthrough_records",
    "default_qbi_passive_passthrough_resource_path",
    "load_qbi_passive_passthrough_resource",
    "validate_qbi_passive_passthrough_resource",
    "weighted_inverse_cdf",
]
