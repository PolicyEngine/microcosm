"""Build and validate provisional evidence resources for QBI v3.

This module owns deterministic estimation logic.  The command-line tool in
``tools/build_us_qbi_v3_evidence.py`` owns restricted-file I/O, input hashing,
and writing the derived JSON resources.  Nothing in this module wires the
evidence into a simulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCF_IMPLICATE_COUNT = 5
SCF_MINIMUM_UNWEIGHTED_N = 30.0
SCF_MARGIN_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
JCT_ZERO_EMPLOYEE_FIRM_SHARE = 0.842

SCF_REQUIRED_COLUMNS = (
    "y1",
    "x42001",
    "x3103",
    "x3104",
    "x3105",
    "x3107",
    "x3111",
    "x3113",
    "x3114",
    "x3119",
    "x3128",
    "x3131",
    "x3132",
    "x3207",
    "x3211",
    "x3213",
    "x3214",
    "x3219",
    "x3228",
    "x3231",
    "x3232",
)

SCF_INCOME_BANDS = (
    "nonpositive",
    "0_to_25k",
    "25k_to_100k",
    "100k_to_250k",
    "250k_to_1m",
    "over_1m",
)
SCF_LEGAL_FORM_GROUPS = (
    "partnership_or_llc",
    "sole_or_informal",
    "s_corporation",
    "other_or_unknown",
)
SCF_INDUSTRY_BINS = (1, 2, 3, 4, 5, 6, 7, 99)
SCF_HEADCOUNT_SIZE_BANDS = (
    "2_to_4",
    "5_to_9",
    "10_to_24",
    "25_to_99",
    "100_plus",
)

_LEGAL_FORM_BY_CODE = {
    1: "partnership_or_llc",
    11: "partnership_or_llc",
    12: "partnership_or_llc",
    15: "partnership_or_llc",
    2: "sole_or_informal",
    40: "sole_or_informal",
    3: "s_corporation",
    4: "other_or_unknown",
    6: "other_or_unknown",
    -7: "other_or_unknown",
}
_MAIN_QBI_PROXY_CODES = frozenset({1, 2, 3, 11, 12, 15, 40})
_STRICT_QBI_PROXY_CODES = frozenset({1, 2, 3, 11})


@dataclass(frozen=True)
class _EstimateSelection:
    """The sample selected by a thin-cell fallback hierarchy."""

    level: str
    frame: pd.DataFrame
    dimensions: Mapping[str, object]


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        invalid = ~np.isfinite(result[column].to_numpy(dtype=np.float64))
        if invalid.any():
            raise ValueError(
                f"SCF column {column!r} contains {int(invalid.sum())} "
                "nonnumeric or nonfinite value(s)."
            )
    return result


def _normalized_scf_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={str(column): str(column).lower() for column in frame}
    )
    duplicate_columns = normalized.columns[normalized.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            "SCF frame has case-insensitive duplicate columns "
            f"{sorted(set(duplicate_columns))}."
        )
    missing = sorted(set(SCF_REQUIRED_COLUMNS) - set(normalized.columns))
    if missing:
        raise ValueError(f"SCF frame is missing required columns {missing}.")
    return _finite_numeric(normalized, SCF_REQUIRED_COLUMNS)


def _income_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=(-np.inf, 0.0, 25_000.0, 100_000.0, 250_000.0, 1_000_000.0, np.inf),
        labels=SCF_INCOME_BANDS,
        right=True,
        include_lowest=True,
    ).astype("object")


def _headcount_size_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=(1.0, 4.0, 9.0, 24.0, 99.0, np.inf),
        labels=SCF_HEADCOUNT_SIZE_BANDS,
        right=True,
    ).astype("object")


def build_scf_business_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Stack the first two actively managed SCF business records.

    Each returned row is one household-business-implicate record.  The point
    estimate weight is ``X42001 / 5``; ``unweighted_n`` elsewhere in this
    module similarly divides pooled record counts by five so the implicates
    are not treated as independent observations.
    """

    source = _normalized_scf_frame(frame)
    y1 = source["y1"].to_numpy(dtype=np.int64)
    if not np.array_equal(y1.astype(np.float64), source["y1"].to_numpy()):
        raise ValueError("SCF Y1 must contain integer identifiers.")
    if pd.Series(y1).duplicated().any():
        raise ValueError("SCF Y1 must be unique across household implicates.")
    household_id = y1 // 10
    implicate = y1 % 10
    expected_implicates = set(range(1, SCF_IMPLICATE_COUNT + 1))
    by_household = pd.DataFrame(
        {"household_id": household_id, "implicate": implicate}
    ).groupby("household_id")["implicate"]
    malformed = [
        int(identifier)
        for identifier, values in by_household
        if set(values.tolist()) != expected_implicates
    ]
    if malformed:
        raise ValueError(
            "SCF Y1 must encode exactly implicates 1-5 for every household; "
            f"malformed household ids include {malformed[:5]}."
        )

    active = source["x3103"].eq(1) & source["x3104"].eq(1) & source["x3105"].ge(1)
    pooled_weight = source["x42001"] / SCF_IMPLICATE_COUNT
    if (pooled_weight <= 0.0).any():
        raise ValueError("SCF X42001 must be positive for every implicate.")

    records: list[pd.DataFrame] = []
    slot_columns = {
        1: {
            "industry_code": "x3107",
            "headcount": "x3111",
            "respondent_works": "x3113",
            "spouse_works": "x3114",
            "legal_form_code": "x3119",
            "ownership_percent_x100": "x3128",
            "gross_receipts": "x3131",
            "whole_net_income": "x3132",
        },
        2: {
            "industry_code": "x3207",
            "headcount": "x3211",
            "respondent_works": "x3213",
            "spouse_works": "x3214",
            "legal_form_code": "x3219",
            "ownership_percent_x100": "x3228",
            "gross_receipts": "x3231",
            "whole_net_income": "x3232",
        },
    }
    for slot, columns in slot_columns.items():
        selected = active & source["x3105"].ge(slot)
        slot_frame = source.loc[selected, list(columns.values())].rename(
            columns={value: key for key, value in columns.items()}
        )
        slot_frame.insert(0, "y1", y1[selected])
        slot_frame.insert(1, "household_id", household_id[selected])
        slot_frame.insert(2, "implicate", implicate[selected])
        slot_frame.insert(3, "business_slot", slot)
        slot_frame.insert(4, "weight", pooled_weight.loc[selected].to_numpy())
        records.append(slot_frame.reset_index(drop=True))

    if not records:
        raise ValueError("SCF frame produced no actively managed business records.")
    business = pd.concat(records, ignore_index=True)
    if business.empty:
        raise ValueError("SCF frame produced no actively managed business records.")

    integer_columns = (
        "industry_code",
        "headcount",
        "respondent_works",
        "spouse_works",
        "legal_form_code",
        "ownership_percent_x100",
    )
    for column in integer_columns:
        values = business[column].to_numpy(dtype=np.float64)
        if not np.array_equal(values, values.astype(np.int64)):
            raise ValueError(f"SCF business field {column!r} must be integer-coded.")
        business[column] = values.astype(np.int64)

    business["headcount"] = business["headcount"].replace(-1, 0)
    if (business["headcount"] < 0).any():
        raise ValueError(
            "SCF business headcount contains an unsupported negative code."
        )
    business["gross_receipts"] = business["gross_receipts"].replace(-1.0, 0.0)
    business["whole_net_income"] = business["whole_net_income"].replace(-1.0, 0.0)
    if (business["gross_receipts"] < 0.0).any():
        raise ValueError("SCF gross receipts contain an unsupported negative value.")
    ownership = business["ownership_percent_x100"]
    if ((ownership < 0) | (ownership > 10_000)).any():
        raise ValueError("SCF ownership percentage must lie between 0 and 10,000.")

    unsupported_forms = sorted(
        set(business["legal_form_code"]) - set(_LEGAL_FORM_BY_CODE)
    )
    if unsupported_forms:
        raise ValueError(
            f"SCF business records contain unsupported legal-form codes "
            f"{unsupported_forms}."
        )
    unsupported_industries = sorted(
        set(business["industry_code"]) - set(SCF_INDUSTRY_BINS)
    )
    if unsupported_industries:
        raise ValueError(
            f"SCF business records contain unsupported industry codes "
            f"{unsupported_industries}."
        )

    business["legal_form_group"] = business["legal_form_code"].map(_LEGAL_FORM_BY_CODE)
    business["owned_net_income"] = business["whole_net_income"] * ownership / 10_000.0
    business["income_band"] = _income_band(business["owned_net_income"])
    business["employer_presence_proxy"] = business["headcount"] > 1
    business["headcount_size_band"] = _headcount_size_band(business["headcount"])
    business["respondent_or_spouse_works"] = business["respondent_works"].eq(
        1
    ) | business["spouse_works"].eq(1)
    business["qbi_positive_proxy"] = business["owned_net_income"].gt(0.0) & business[
        "legal_form_code"
    ].isin(_MAIN_QBI_PROXY_CODES)
    business["qbi_positive_strict_form_proxy"] = business["owned_net_income"].gt(
        0.0
    ) & business["legal_form_code"].isin(_STRICT_QBI_PROXY_CODES)
    return business.reset_index(drop=True)


def weighted_inverse_cdf(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] = SCF_MARGIN_QUANTILES,
) -> dict[str, float]:
    """Return weighted inverse-CDF quantiles with stable tie handling."""

    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if value_array.shape != weight_array.shape:
        raise ValueError("Weighted quantile values and weights must have equal shape.")
    if value_array.ndim != 1:
        raise ValueError(
            "Weighted quantile values and weights must be one-dimensional."
        )
    if (
        not np.isfinite(value_array).all()
        or not np.isfinite(weight_array).all()
        or (weight_array <= 0.0).any()
    ):
        raise ValueError(
            "Weighted quantiles require finite values and positive weights."
        )
    if value_array.size == 0:
        raise ValueError("Weighted quantiles require at least one observation.")
    if (
        not np.isfinite(probability_array).all()
        or (probability_array < 0.0).any()
        or (probability_array > 1.0).any()
    ):
        raise ValueError("Weighted quantile probabilities must lie in [0, 1].")

    order = np.argsort(value_array, kind="stable")
    sorted_values = value_array[order]
    cumulative = np.cumsum(weight_array[order])
    targets = probability_array * cumulative[-1]
    positions = np.searchsorted(cumulative, targets, side="left")
    positions = np.minimum(positions, len(sorted_values) - 1)
    return {
        _quantile_name(probability): float(sorted_values[position])
        for probability, position in zip(probability_array, positions, strict=True)
    }


def _quantile_name(probability: float) -> str:
    return f"q{int(round(probability * 100)):02d}"


def _unweighted_n(frame: pd.DataFrame) -> float:
    return float(len(frame) / SCF_IMPLICATE_COUNT)


def _counts(frame: pd.DataFrame) -> dict[str, float | int]:
    employer = frame.loc[frame["employer_presence_proxy"]]
    return {
        "pooled_record_count": int(len(frame)),
        "implicate_adjusted_unweighted_n": _unweighted_n(frame),
        "weighted_business_interests": float(frame["weight"].sum()),
        "employer_proxy_pooled_record_count": int(len(employer)),
        "employer_proxy_implicate_adjusted_unweighted_n": _unweighted_n(employer),
        "weighted_employer_proxy_business_interests": float(employer["weight"].sum()),
    }


def _candidate_sample(
    records: pd.DataFrame,
    *,
    income_band: str,
    legal_form_group: str,
    industry_code: int,
    employer_only: bool,
    minimum_n: float,
) -> _EstimateSelection:
    base = (
        records["employer_presence_proxy"]
        if employer_only
        else pd.Series(True, index=records.index)
    )
    candidates = (
        (
            "exact",
            base
            & records["income_band"].eq(income_band)
            & records["legal_form_group"].eq(legal_form_group)
            & records["industry_code"].eq(industry_code),
            {
                "income_band": income_band,
                "legal_form_group": legal_form_group,
                "industry_code": industry_code,
            },
        ),
        (
            "income_form",
            base
            & records["income_band"].eq(income_band)
            & records["legal_form_group"].eq(legal_form_group),
            {
                "income_band": income_band,
                "legal_form_group": legal_form_group,
                "industry_code": "all",
            },
        ),
        (
            "form",
            base & records["legal_form_group"].eq(legal_form_group),
            {
                "income_band": "all",
                "legal_form_group": legal_form_group,
                "industry_code": "all",
            },
        ),
        (
            "all",
            base,
            {
                "income_band": "all",
                "legal_form_group": "all",
                "industry_code": "all",
            },
        ),
    )
    for index, (level, mask, dimensions) in enumerate(candidates):
        sample = records.loc[mask]
        if _unweighted_n(sample) >= minimum_n or index == len(candidates) - 1:
            if sample.empty:
                raise ValueError(
                    "SCF fallback hierarchy reached an empty global sample."
                )
            return _EstimateSelection(level, sample, dimensions)
    raise AssertionError("SCF fallback hierarchy did not select a sample.")


def _presence_estimate(selection: _EstimateSelection) -> dict[str, object]:
    weights = selection.frame["weight"].to_numpy(dtype=np.float64)
    present = selection.frame["employer_presence_proxy"].to_numpy(dtype=np.float64)
    return {
        "probability_headcount_gt_1": float(np.sum(weights * present) / weights.sum()),
        "estimate_level": selection.level,
        "source_dimensions": dict(selection.dimensions),
        "source_counts": _counts(selection.frame),
    }


def _size_estimate(selection: _EstimateSelection) -> dict[str, object]:
    denominator = float(selection.frame["weight"].sum())
    shares = {}
    for band in SCF_HEADCOUNT_SIZE_BANDS:
        weight = selection.frame.loc[
            selection.frame["headcount_size_band"].eq(band), "weight"
        ].sum()
        shares[band] = float(weight / denominator)
    return {
        "conditional_on": "headcount_gt_1",
        "shares": shares,
        "estimate_level": selection.level,
        "source_dimensions": dict(selection.dimensions),
        "source_counts": _counts(selection.frame),
    }


def _comparison(records: pd.DataFrame, flag: str) -> dict[str, float | int]:
    sample = records.loc[records[flag]]
    if sample.empty:
        raise ValueError(f"SCF comparison flag {flag!r} selected no records.")
    weights = sample["weight"].to_numpy(dtype=np.float64)
    zero_proxy = ~sample["employer_presence_proxy"].to_numpy(dtype=bool)
    share = float(weights[zero_proxy].sum() / weights.sum())
    return {
        "pooled_record_count": int(len(sample)),
        "implicate_adjusted_unweighted_n": _unweighted_n(sample),
        "weighted_business_interests": float(weights.sum()),
        "headcount_le_1_share": share,
        "difference_from_jct_percentage_points": float(
            100.0 * (share - JCT_ZERO_EMPLOYEE_FIRM_SHARE)
        ),
    }


def _profit_margin_cells(
    records: pd.DataFrame,
    *,
    minimum_n: float,
) -> list[dict[str, object]]:
    positive = records.loc[
        records["owned_net_income"].gt(0.0) & records["gross_receipts"].gt(0.0)
    ].copy()
    positive["profit_margin"] = (
        positive["whole_net_income"] / positive["gross_receipts"]
    )
    if positive.empty:
        raise ValueError("SCF records contain no positive profit-margin sample.")

    cells: list[dict[str, object]] = []
    for legal_form_group in SCF_LEGAL_FORM_GROUPS:
        form_sample = positive.loc[positive["legal_form_group"].eq(legal_form_group)]
        for industry_code in SCF_INDUSTRY_BINS:
            exact = form_sample.loc[form_sample["industry_code"].eq(industry_code)]
            candidates = (
                (
                    "exact",
                    exact,
                    {
                        "legal_form_group": legal_form_group,
                        "industry_code": industry_code,
                    },
                ),
                (
                    "form",
                    form_sample,
                    {
                        "legal_form_group": legal_form_group,
                        "industry_code": "all",
                    },
                ),
                (
                    "all",
                    positive,
                    {"legal_form_group": "all", "industry_code": "all"},
                ),
            )
            selection: _EstimateSelection | None = None
            for index, (level, sample, dimensions) in enumerate(candidates):
                if _unweighted_n(sample) >= minimum_n or index == len(candidates) - 1:
                    selection = _EstimateSelection(level, sample, dimensions)
                    break
            if selection is None or selection.frame.empty:
                raise ValueError("SCF profit-margin fallback selected no records.")
            margins = selection.frame["profit_margin"].to_numpy(dtype=np.float64)
            weights = selection.frame["weight"].to_numpy(dtype=np.float64)
            cells.append(
                {
                    "legal_form_group": legal_form_group,
                    "industry_code": industry_code,
                    "requested_counts": {
                        "pooled_record_count": int(len(exact)),
                        "implicate_adjusted_unweighted_n": _unweighted_n(exact),
                        "weighted_business_interests": float(exact["weight"].sum()),
                    },
                    "estimate_level": selection.level,
                    "source_dimensions": dict(selection.dimensions),
                    "source_counts": {
                        "pooled_record_count": int(len(selection.frame)),
                        "implicate_adjusted_unweighted_n": _unweighted_n(
                            selection.frame
                        ),
                        "weighted_business_interests": float(weights.sum()),
                    },
                    "quantiles": weighted_inverse_cdf(
                        margins, weights, SCF_MARGIN_QUANTILES
                    ),
                    "source_minimum": float(margins.min()),
                    "source_maximum": float(margins.max()),
                }
            )
    return cells


def build_qbi_employer_structure_resource(
    frame: pd.DataFrame,
    *,
    provenance: Mapping[str, object],
    minimum_unweighted_n: float = SCF_MINIMUM_UNWEIGHTED_N,
) -> dict[str, object]:
    """Build the provisional SCF employer-structure evidence resource."""

    if not math.isfinite(minimum_unweighted_n) or minimum_unweighted_n <= 0.0:
        raise ValueError("SCF minimum unweighted n must be positive and finite.")
    records = build_scf_business_records(frame)
    cells: list[dict[str, object]] = []
    for income_band in SCF_INCOME_BANDS:
        for legal_form_group in SCF_LEGAL_FORM_GROUPS:
            for industry_code in SCF_INDUSTRY_BINS:
                exact = records.loc[
                    records["income_band"].eq(income_band)
                    & records["legal_form_group"].eq(legal_form_group)
                    & records["industry_code"].eq(industry_code)
                ]
                presence = _candidate_sample(
                    records,
                    income_band=income_band,
                    legal_form_group=legal_form_group,
                    industry_code=industry_code,
                    employer_only=False,
                    minimum_n=minimum_unweighted_n,
                )
                size = _candidate_sample(
                    records,
                    income_band=income_band,
                    legal_form_group=legal_form_group,
                    industry_code=industry_code,
                    employer_only=True,
                    minimum_n=minimum_unweighted_n,
                )
                cells.append(
                    {
                        "income_band": income_band,
                        "legal_form_group": legal_form_group,
                        "industry_code": industry_code,
                        "requested_counts": _counts(exact),
                        "employer_presence": _presence_estimate(presence),
                        "headcount_size_distribution": _size_estimate(size),
                    }
                )

    payload: dict[str, object] = {
        "schema_version": 1,
        "resource_id": "qbi_employer_structure_v1",
        "provisional": True,
        "source": {
            "survey": "2022 Survey of Consumer Finances full public data",
            "survey_year": 2022,
            "business_flow_year": 2021,
            "record_scope": (
                "First and second detailed actively managed household businesses"
            ),
            "variables": {
                "business_count": "X3105",
                "weight": "X42001",
                "industry": ["X3107", "X3207"],
                "headcount": ["X3111", "X3211"],
                "legal_form": ["X3119", "X3219"],
                "ownership_percent_x100": ["X3128", "X3228"],
                "gross_receipts": ["X3131", "X3231"],
                "whole_business_net_income": ["X3132", "X3232"],
                "respondent_or_spouse_works": [
                    "X3113",
                    "X3114",
                    "X3213",
                    "X3214",
                ],
            },
            "codebook": {
                "url": ("https://www.federalreserve.gov/econres/files/codebk2022.txt"),
                "implicate_convention_lines": "622-655",
                "weight_definition_lines": "1865-1873, 2792-2796",
                "business_variable_lines": "11699-12534",
            },
        },
        "methodology": {
            "point_estimate_weight": "X42001 divided by 5",
            "implicate_count": SCF_IMPLICATE_COUNT,
            "implicate_note": (
                "Point estimates pool all five implicates with record weight "
                "X42001/5, following the SCF simple-statistic convention. "
                "No replicate-weight or multiple-imputation sampling-error "
                "estimate is claimed, and implicates are not treated as "
                "independent observations."
            ),
            "income_measure": (
                "Whole-business pretax net income multiplied by the household "
                "ownership percentage; exact code -1 ('Nothing') is zero."
            ),
            "employer_presence_proxy": "headcount greater than 1",
            "employer_proxy_limitation": (
                "SCF headcount includes owners, family members, unpaid workers, "
                "and all paid full- and part-time workers, so headcount>1 is an "
                "upper-bound employer-presence proxy."
            ),
            "minimum_implicate_adjusted_unweighted_n": minimum_unweighted_n,
            "presence_collapse_order": [
                "income_form_industry",
                "income_form",
                "form",
                "all",
            ],
            "size_collapse_order": [
                "income_form_industry_among_headcount_gt_1",
                "income_form_among_headcount_gt_1",
                "form_among_headcount_gt_1",
                "all_among_headcount_gt_1",
            ],
            "size_collapse_note": (
                "Employer size is collapsed independently on its headcount>1 "
                "denominator so a cell with 30 total businesses but few employer "
                "proxies does not retain a thin size distribution."
            ),
            "headcount_disclosure": (
                "Counts above 10 are rounded to the nearest 5 and top-coded at 5,000."
            ),
        },
        "dimensions": {
            "income_bands": [
                {"id": "nonpositive", "definition": "owned net income <= 0"},
                {"id": "0_to_25k", "definition": "0 < owned net income <= 25,000"},
                {
                    "id": "25k_to_100k",
                    "definition": "25,000 < owned net income <= 100,000",
                },
                {
                    "id": "100k_to_250k",
                    "definition": "100,000 < owned net income <= 250,000",
                },
                {
                    "id": "250k_to_1m",
                    "definition": "250,000 < owned net income <= 1,000,000",
                },
                {"id": "over_1m", "definition": "owned net income > 1,000,000"},
            ],
            "legal_form_groups": [
                {"id": "partnership_or_llc", "public_codes": [1, 11]},
                {"id": "sole_or_informal", "public_codes": [2, 40]},
                {"id": "s_corporation", "public_codes": [3]},
                {"id": "other_or_unknown", "public_codes": [4, -7]},
            ],
            "industry_bins": [
                {"code": 1, "summary": "agriculture plus veterinary/landscaping"},
                {"code": 2, "summary": "mining and construction"},
                {"code": 3, "summary": "manufacturing plus selected publishing"},
                {"code": 4, "summary": "wholesale/retail plus food services"},
                {
                    "code": 5,
                    "summary": (
                        "finance/real estate plus selected information, "
                        "business support, rental, and repair"
                    ),
                },
                {"code": 6, "summary": "remaining private service industries"},
                {"code": 7, "summary": "public administration and military"},
                {
                    "code": 99,
                    "summary": (
                        "industry suppressed for very-high-value businesses; "
                        "not an economic industry"
                    ),
                },
            ],
            "headcount_size_bands": [
                {"id": "2_to_4", "definition": "2-4 people"},
                {"id": "5_to_9", "definition": "5-9 people"},
                {"id": "10_to_24", "definition": "10-24 people"},
                {"id": "25_to_99", "definition": "25-99 people"},
                {"id": "100_plus", "definition": "100 or more people"},
            ],
        },
        "external_anchor": {
            "source": (
                "Joint Committee on Taxation, Data Related to Deductions "
                "Claimed Under Code Section 199A, Table 5"
            ),
            "source_url": (
                "https://www.warren.senate.gov/imo/media/doc/jct_report_on_199a.pdf"
            ),
            "tax_year": 2022,
            "zero_employee_firm_share": JCT_ZERO_EMPLOYEE_FIRM_SHARE,
            "zero_employee_deduction_dollar_share": 0.357,
            "sole_proprietor_zero_employee_share_statement": "more than 95%",
            "partnership_zero_employee_share_statement": "more than 80%",
            "scf_comparison": {
                "main_proxy_including_informal_code_40": _comparison(
                    records, "qbi_positive_proxy"
                ),
                "strict_form_sensitivity_excluding_code_40": _comparison(
                    records, "qbi_positive_strict_form_proxy"
                ),
            },
            "definition_gaps": [
                (
                    "JCT counts deduction-generating tax firms with zero W-2 "
                    "employees; SCF observes household business interests and "
                    "only whether reported headcount exceeds one."
                ),
                (
                    "SCF headcount includes owners, family, unpaid workers, and "
                    "paid workers, making its headcount<=1 share a lower bound "
                    "on a zero-W-2-employer share."
                ),
                (
                    "JCT uses tax year 2022 and actual QBID-generating firms; "
                    "SCF survey-year 2022 reports calendar-2021 pretax income "
                    "without SSTB, aggregation, carryforward, taxable-income, "
                    "or deduction eligibility rules."
                ),
                (
                    "SCF details only the first/largest two actively managed "
                    "businesses and cannot deduplicate a firm owned by households "
                    "observed separately."
                ),
            ],
        },
        "cells": cells,
        "profit_margin_quantiles": {
            "definition": (
                "Whole-business pretax net income divided by whole-business "
                "gross receipts, among positive owned net income and receipts."
            ),
            "ownership_note": (
                "The ownership percentage cancels from the numerator and "
                "denominator of the margin."
            ),
            "support_note": (
                "Margins are empirical and uncapped; SCF permits reported net "
                "income above gross receipts."
            ),
            "quantile_method": (
                "weighted inverse CDF: smallest margin whose cumulative "
                "X42001/5 weight reaches the requested probability"
            ),
            "probabilities": list(SCF_MARGIN_QUANTILES),
            "minimum_implicate_adjusted_unweighted_n": minimum_unweighted_n,
            "collapse_order": ["form_industry", "form", "all"],
            "cells": _profit_margin_cells(records, minimum_n=minimum_unweighted_n),
        },
        "provenance": dict(provenance),
        "judgment_calls": [
            (
                "Income bands use the household-owned share of whole-business "
                "net income to align with recipient QBI; whole-business income "
                "would be a more direct predictor of firm headcount."
            ),
            (
                "Code 40 ('not a formal business type') is grouped with sole "
                "proprietorships in the main comparison; a strict sensitivity "
                "excludes it."
            ),
            (
                "Band boundaries, minimum n=30, the nested collapse hierarchy, "
                "independent employer-size collapse, and the five-point margin "
                "quantile grid are modeling choices pending supervisor review."
            ),
        ],
    }
    validate_qbi_employer_structure_resource(payload)
    return payload


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    return value


def _list(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a JSON array.")
    return value


def _finite_probability(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be finite and lie in [0, 1].")
    return result


def validate_qbi_employer_structure_resource(
    payload: Mapping[str, object],
) -> None:
    """Validate the committed SCF evidence-resource contract."""

    root = _mapping(payload, "QBI employer resource")
    required = {
        "schema_version",
        "resource_id",
        "provisional",
        "source",
        "methodology",
        "dimensions",
        "external_anchor",
        "cells",
        "profit_margin_quantiles",
        "provenance",
        "judgment_calls",
    }
    if set(root) != required:
        raise ValueError(
            "QBI employer resource top-level keys must be exactly "
            f"{sorted(required)}; got {sorted(root)}."
        )
    if root["schema_version"] != 1:
        raise ValueError("QBI employer resource schema_version must equal 1.")
    if root["resource_id"] != "qbi_employer_structure_v1":
        raise ValueError("QBI employer resource_id is not recognized.")
    if root["provisional"] is not True:
        raise ValueError("QBI employer resource must remain provisional.")
    _mapping(root["source"], "source")
    methodology = _mapping(root["methodology"], "methodology")
    minimum_n = methodology.get("minimum_implicate_adjusted_unweighted_n")
    if (
        isinstance(minimum_n, bool)
        or not isinstance(minimum_n, (int, float))
        or not math.isfinite(float(minimum_n))
        or float(minimum_n) <= 0.0
    ):
        raise ValueError("Employer methodology minimum n must be positive.")
    _mapping(root["dimensions"], "dimensions")
    _mapping(root["provenance"], "provenance")
    if not _list(root["judgment_calls"], "judgment_calls"):
        raise ValueError("Employer resource judgment_calls must be nonempty.")

    cells = _list(root["cells"], "cells")
    expected_cell_count = (
        len(SCF_INCOME_BANDS) * len(SCF_LEGAL_FORM_GROUPS) * len(SCF_INDUSTRY_BINS)
    )
    if len(cells) != expected_cell_count:
        raise ValueError(
            f"Employer resource must carry {expected_cell_count} cells; "
            f"got {len(cells)}."
        )
    seen: set[tuple[object, object, object]] = set()
    for index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, f"cells[{index}]")
        key = (
            cell.get("income_band"),
            cell.get("legal_form_group"),
            cell.get("industry_code"),
        )
        if key in seen:
            raise ValueError(f"Employer resource duplicates cell {key}.")
        seen.add(key)
        if key[0] not in SCF_INCOME_BANDS:
            raise ValueError(f"Employer cell has unknown income band {key[0]!r}.")
        if key[1] not in SCF_LEGAL_FORM_GROUPS:
            raise ValueError(f"Employer cell has unknown legal form {key[1]!r}.")
        if key[2] not in SCF_INDUSTRY_BINS:
            raise ValueError(f"Employer cell has unknown industry {key[2]!r}.")
        _mapping(cell.get("requested_counts"), f"cells[{index}].requested_counts")
        presence = _mapping(
            cell.get("employer_presence"), f"cells[{index}].employer_presence"
        )
        _finite_probability(
            presence.get("probability_headcount_gt_1"),
            f"cells[{index}].employer_presence.probability",
        )
        size = _mapping(
            cell.get("headcount_size_distribution"),
            f"cells[{index}].headcount_size_distribution",
        )
        shares = _mapping(size.get("shares"), f"cells[{index}].shares")
        if set(shares) != set(SCF_HEADCOUNT_SIZE_BANDS):
            raise ValueError(f"Employer cell {key} has malformed size bands.")
        values = [
            _finite_probability(shares[band], f"cells[{index}].shares.{band}")
            for band in SCF_HEADCOUNT_SIZE_BANDS
        ]
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Employer cell {key} size shares do not sum to 1.")

    external = _mapping(root["external_anchor"], "external_anchor")
    _finite_probability(
        external.get("zero_employee_firm_share"),
        "external_anchor.zero_employee_firm_share",
    )
    comparisons = _mapping(
        external.get("scf_comparison"), "external_anchor.scf_comparison"
    )
    for name, raw_comparison in comparisons.items():
        comparison = _mapping(raw_comparison, f"external_anchor.scf_comparison.{name}")
        _finite_probability(
            comparison.get("headcount_le_1_share"),
            f"external_anchor.scf_comparison.{name}.headcount_le_1_share",
        )

    margins = _mapping(root["profit_margin_quantiles"], "profit_margin_quantiles")
    margin_cells = _list(margins.get("cells"), "profit_margin_quantiles.cells")
    expected_margin_count = len(SCF_LEGAL_FORM_GROUPS) * len(SCF_INDUSTRY_BINS)
    if len(margin_cells) != expected_margin_count:
        raise ValueError(
            f"Profit-margin resource must carry {expected_margin_count} cells."
        )
    margin_seen: set[tuple[object, object]] = set()
    quantile_names = [_quantile_name(q) for q in SCF_MARGIN_QUANTILES]
    for index, raw_cell in enumerate(margin_cells):
        cell = _mapping(raw_cell, f"profit_margin_quantiles.cells[{index}]")
        key = (cell.get("legal_form_group"), cell.get("industry_code"))
        if key in margin_seen:
            raise ValueError(f"Profit-margin resource duplicates cell {key}.")
        margin_seen.add(key)
        quantiles = _mapping(
            cell.get("quantiles"),
            f"profit_margin_quantiles.cells[{index}].quantiles",
        )
        if list(quantiles) != quantile_names:
            raise ValueError(
                f"Profit-margin cell {key} must carry ordered quantiles "
                f"{quantile_names}."
            )
        values = []
        for name in quantile_names:
            value = quantiles[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Profit-margin cell {key} {name} is not numeric.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Profit-margin cell {key} {name} is not finite.")
            values.append(value)
        if values != sorted(values):
            raise ValueError(f"Profit-margin cell {key} quantiles are not monotone.")
