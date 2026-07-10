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

from populace.calibrate import relative_error_loss

__all__ = [
    "AGGREGATE_RECIDS",
    "SYNTHETIC_RECID_START",
    "PufAggregateDisaggregationSpec",
    "audit_puf_aggregate_disaggregation",
    "compute_aggregate_eligibility_scores",
    "derive_puf_policyengine_variables",
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
_DIVIDEND_INVARIANT_ATOL = 1e-9
_DIVIDEND_RAW_AMOUNT_COLUMNS = frozenset({"E00600", "E00650"})
_DIVIDEND_COMPONENT_AMOUNT_COLUMNS = (
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
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
_SOURCE_FIELD_ATTRIBUTES = {
    "E00100": "adjusted_gross_income",
    "E00200": "employment_income",
    "E00300": "taxable_interest_income",
    "E00400": "tax_exempt_interest_income",
    "E00600": "ordinary_dividends",
    "E00650": "qualified_dividends",
    "E00900": "business_net_profits",
    "E01100": "capital_gains_distributions",
    "E01400": "ira_distributions",
    "E01500": "total_pension_income",
    "E01700": "taxable_pension_income",
    "E02300": "unemployment_compensation",
    "E02400": "total_social_security",
    "E02500": "taxable_social_security",
    "E03210": "student_loan_interest",
    "E17500": "medical_expense_deduction",
    "E18400": "state_income_tax_paid",
    "E18500": "real_estate_taxes_paid",
    "E19200": "mortgage_interest_paid",
    "E19800": "charitable_cash_contributions",
    "E20100": "charitable_noncash_contributions",
    "P22250": "short_term_capital_gains",
    "P23250": "long_term_capital_gains",
    "E24515": "unrecaptured_section_1250_gain",
    "E24518": "collectibles_capital_gains",
    "E26270": "partnership_and_s_corp_income",
    "E87521": "american_opportunity_credit",
}
_COMBINED_SOURCE_FIELDS = {
    "capital_gains_proxy": ("P22250", "P23250", "E01100"),
    "charitable_contributions": ("E19800", "E20100"),
}

_DEFAULT_PUF_POLICYENGINE_VARIABLES = {
    "ordinary_dividend_source": "E00600",
    "qualified_dividend_source": "E00650",
    "qualified_dividend_output": "qualified_dividend_income",
    "non_qualified_dividend_output": "non_qualified_dividend_income",
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


def derive_puf_policyengine_variables(
    puf: pd.DataFrame,
    *,
    ordinary_dividend_source: str = "E00600",
    qualified_dividend_source: str = "E00650",
    qualified_dividend_output: str = "qualified_dividend_income",
    non_qualified_dividend_output: str = "non_qualified_dividend_income",
    qualified_tuition_primary_source: str | None = None,
    qualified_tuition_optional_source: str | None = None,
    qualified_tuition_output: str = "qualified_tuition_expenses",
) -> pd.DataFrame:
    """Translate raw IRS PUF columns into PolicyEngine input variables."""

    _require_columns(puf, [ordinary_dividend_source, qualified_dividend_source])
    result = puf.copy()
    ordinary = _numeric_series(result[ordinary_dividend_source])
    qualified = _numeric_series(result[qualified_dividend_source])
    _assert_dividend_source_invariant(
        ordinary,
        qualified,
        ordinary_source=ordinary_dividend_source,
        qualified_source=qualified_dividend_source,
    )

    result[qualified_dividend_output] = qualified
    result[non_qualified_dividend_output] = ordinary - qualified
    if qualified_tuition_primary_source is not None:
        _require_columns(result, [qualified_tuition_primary_source])
        tuition = _numeric_series(result[qualified_tuition_primary_source]).clip(
            lower=0.0
        )
        if (
            qualified_tuition_optional_source is not None
            and qualified_tuition_optional_source in result
        ):
            optional = _numeric_series(result[qualified_tuition_optional_source]).clip(
                lower=0.0
            )
            tuition = pd.Series(
                np.maximum(tuition.to_numpy(), optional.to_numpy()),
                index=result.index,
                dtype="float64",
            )
        result[qualified_tuition_output] = tuition
    return result


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
    puf = _derive_default_puf_policyengine_variables_if_available(puf)

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
    amount_columns = _get_disaggregation_amount_columns(puf.columns)
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
    result = pd.concat([regular, synthetic_df], ignore_index=True)
    result = _reconcile_puf_dividend_columns_from_components(result)
    return _reconcile_puf_qualified_tuition_from_sources(result)


def audit_puf_aggregate_disaggregation(
    puf: pd.DataFrame,
    *,
    seed: int = 42,
    spec: PufAggregateDisaggregationSpec | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Measure source-level recovery from replacing PUF aggregate rows.

    The target surface here is the raw PUF itself, including the four IRS
    aggregate disclosure rows. The old source support drops those rows; the new
    support replaces them with synthetic records. This audit is therefore a
    deterministic source-reconstruction check, not a calibrated release
    benchmark.
    """

    spec = spec or load_default_puf_aggregate_disaggregation_spec()
    _require_columns(puf, ["RECID", "S006", "E00100"])
    puf = _derive_default_puf_policyengine_variables_if_available(puf)

    aggregate_mask = puf["RECID"].isin(spec.aggregate_recids)
    _require_raw_aggregate_rows(puf, spec)
    regular = puf.loc[~aggregate_mask].copy()
    result = disaggregate_puf_aggregate_records(puf, seed=seed, spec=spec)
    synthetic = result.loc[result["RECID"] >= spec.synthetic_recid_start].copy()
    amount_columns = _audit_amount_columns(puf, requested_columns=columns)

    target_totals = {column: _weighted_sum(puf, column) for column in amount_columns}
    old_totals = {column: _weighted_sum(regular, column) for column in amount_columns}
    disaggregated_totals = {
        column: _weighted_sum(result, column) for column in amount_columns
    }

    return {
        "source": spec.source,
        "method": "donor_template_calibration",
        "seed": int(seed),
        "forbes_top_tail": bool(spec.forbes_top_tail),
        "raw_rows": int(len(puf)),
        "regular_rows": int(len(regular)),
        "raw_aggregate_rows": int(aggregate_mask.sum()),
        "result_rows": int(len(result)),
        "synthetic_rows": int(len(synthetic)),
        "aggregate_rows_after": int(result["RECID"].isin(spec.aggregate_recids).sum()),
        "raw_aggregate_s006": _s006_sum(puf.loc[aggregate_mask]),
        "synthetic_s006": _s006_sum(synthetic),
        "raw_aggregate_weight": _s006_weight_total(puf.loc[aggregate_mask]),
        "synthetic_weight": _s006_weight_total(synthetic),
        "synthetic_mars_counts": _value_counts(synthetic, "MARS"),
        "amount_column_count": int(len(amount_columns)),
        "source_reconstruction_loss": {
            "old_drop_aggregate": _source_loss(old_totals, target_totals),
            "disaggregated": _source_loss(disaggregated_totals, target_totals),
        },
        "field_totals": _audit_field_totals(
            puf=puf,
            old=regular,
            disaggregated=result,
        ),
        "bucket_summaries": _audit_bucket_summaries(
            puf=puf,
            disaggregated=result,
            amount_columns=amount_columns,
            spec=spec,
        ),
    }


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


def _source_loss(
    estimates: dict[str, float],
    targets: dict[str, float],
) -> dict[str, Any]:
    errors = [
        {
            "source_column": column,
            "target_total": _json_float(target),
            "estimate_total": _json_float(estimates.get(column, 0.0)),
            "relative_error": _json_float(
                _relative_error(estimates.get(column, 0.0), target)
            ),
        }
        for column, target in targets.items()
    ]
    if not errors:
        return {
            "loss": 0.0,
            "loss_formula": (
                "mean(min(abs((estimate - target) / max(abs(target), 1)), 10))"
            ),
            "n_columns": 0,
            "within_10pct": 1.0,
            "max_abs_relative_error": 0.0,
            "max_abs_relative_error_column": None,
            "worst_columns": [],
        }

    ranked = sorted(
        errors,
        key=lambda error: abs(error["relative_error"]),
        reverse=True,
    )
    worst = ranked[0]
    return {
        "loss": _json_float(
            relative_error_loss(
                np.asarray([error["estimate_total"] for error in errors]),
                np.asarray([error["target_total"] for error in errors]),
            )
        ),
        "loss_formula": (
            "mean(min(abs((estimate - target) / max(abs(target), 1)), 10))"
        ),
        "n_columns": int(len(errors)),
        "within_10pct": _json_float(
            float(np.mean([abs(error["relative_error"]) <= 0.10 for error in errors]))
        ),
        "max_abs_relative_error": _json_float(abs(worst["relative_error"])),
        "max_abs_relative_error_column": worst["source_column"],
        "worst_columns": ranked[:20],
    }


def _audit_field_totals(
    *,
    puf: pd.DataFrame,
    old: pd.DataFrame,
    disaggregated: pd.DataFrame,
) -> list[dict[str, Any]]:
    fields: list[tuple[str, tuple[str, ...]]] = [
        (attribute, (column,))
        for column, attribute in _SOURCE_FIELD_ATTRIBUTES.items()
        if column in puf.columns
    ]
    fields.extend(
        (attribute, tuple(column for column in columns if column in puf.columns))
        for attribute, columns in _COMBINED_SOURCE_FIELDS.items()
        if any(column in puf.columns for column in columns)
    )

    results: list[dict[str, Any]] = []
    for attribute, columns in fields:
        target_total = _weighted_sum_columns(puf, columns)
        old_total = _weighted_sum_columns(old, columns)
        disaggregated_total = _weighted_sum_columns(disaggregated, columns)
        row: dict[str, Any] = {
            "attribute": attribute,
            "source_columns": list(columns),
            "target_total": _json_float(target_total),
            "old_drop_aggregate_total": _json_float(old_total),
            "disaggregated_total": _json_float(disaggregated_total),
            "aggregate_row_total": _json_float(target_total - old_total),
            "recovered_total": _json_float(disaggregated_total - old_total),
            "old_relative_error": _json_float(_relative_error(old_total, target_total)),
            "disaggregated_relative_error": _json_float(
                _relative_error(disaggregated_total, target_total)
            ),
        }
        if len(columns) == 1:
            row["source_column"] = columns[0]
        results.append(row)
    return results


def _audit_bucket_summaries(
    *,
    puf: pd.DataFrame,
    disaggregated: pd.DataFrame,
    amount_columns: list[str],
    spec: PufAggregateDisaggregationSpec,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    next_recid = spec.synthetic_recid_start
    key_columns = [
        column
        for column in ("E00100", "P23250", "P22250", "E26270", "E00900")
        if column in amount_columns
    ]

    for recid in spec.aggregate_recids:
        matching = puf.loc[puf["RECID"] == recid]
        if matching.empty:
            continue
        row = matching.iloc[0]
        pop_weight, _, _ = _get_bucket_targets(row)
        total_s006 = int(round(_finite_float(row["S006"])))
        n_synthetic = min(_choose_n_synthetic(pop_weight), max(total_s006, 1))
        start = next_recid
        end = start + n_synthetic
        next_recid = end
        bucket = disaggregated.loc[
            (disaggregated["RECID"] >= start) & (disaggregated["RECID"] < end)
        ]

        column_deltas = []
        for column in amount_columns:
            target = pop_weight * _finite_float(row.get(column, 0.0))
            achieved = _weighted_sum(bucket, column)
            column_deltas.append(
                {
                    "source_column": column,
                    "target_total": _json_float(target),
                    "achieved_total": _json_float(achieved),
                    "delta": _json_float(achieved - target),
                }
            )
        ranked = sorted(
            column_deltas,
            key=lambda item: abs(item["delta"]),
            reverse=True,
        )
        worst = ranked[0] if ranked else None
        summaries.append(
            {
                "recid": int(recid),
                "description": spec.buckets[recid].description,
                "synthetic_recid_start": int(start),
                "synthetic_recid_end_exclusive": int(end),
                "synthetic_rows": int(len(bucket)),
                "target_s006": _json_float(_finite_float(row["S006"])),
                "synthetic_s006": _s006_sum(bucket),
                "target_weight": _json_float(pop_weight),
                "synthetic_weight": _s006_weight_total(bucket),
                "max_abs_weighted_total_delta": _json_float(
                    abs(worst["delta"]) if worst else 0.0
                ),
                "max_delta_column": worst["source_column"] if worst else None,
                "key_column_totals": [
                    _audit_column_total(
                        source_column=column,
                        target_total=pop_weight * _finite_float(row.get(column, 0.0)),
                        achieved_total=_weighted_sum(bucket, column),
                    )
                    for column in key_columns
                ],
                "largest_column_deltas": ranked[:10],
            }
        )
    return summaries


def _require_raw_aggregate_rows(
    puf: pd.DataFrame,
    spec: PufAggregateDisaggregationSpec,
) -> None:
    found = sorted(
        set(pd.to_numeric(puf["RECID"], errors="coerce").dropna().astype(int).tolist())
        & set(spec.aggregate_recids)
    )
    expected = sorted(spec.aggregate_recids)
    if found != expected:
        raise ValueError(
            "PUF aggregate disaggregation audit requires the raw IRS aggregate "
            f"RECIDs {expected}; found {found}. Run it on the raw PUF, not an "
            "already-disaggregated source file."
        )


def _audit_amount_columns(
    puf: pd.DataFrame,
    *,
    requested_columns: list[str] | None,
) -> list[str]:
    if requested_columns is None:
        amount_columns = _get_amount_columns(puf.columns)
    else:
        missing = [column for column in requested_columns if column not in puf.columns]
        if missing:
            raise ValueError(f"Requested PUF audit columns are missing: {missing}.")
        non_amount = [
            column
            for column in requested_columns
            if not _AMOUNT_COLUMN_PATTERN.match(column)
        ]
        if non_amount:
            raise ValueError(
                "Requested PUF audit columns must be PUF amount columns matching "
                f"{_AMOUNT_COLUMN_PATTERN.pattern!r}; got {non_amount}."
            )
        amount_columns = list(requested_columns)
    if not amount_columns:
        raise ValueError("PUF aggregate disaggregation audit has no amount columns.")
    return amount_columns


def _audit_column_total(
    *,
    source_column: str,
    target_total: float,
    achieved_total: float,
) -> dict[str, Any]:
    return {
        "source_column": source_column,
        "target_total": _json_float(target_total),
        "achieved_total": _json_float(achieved_total),
        "delta": _json_float(achieved_total - target_total),
        "relative_error": _json_float(_relative_error(achieved_total, target_total)),
    }


def _weighted_sum_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> float:
    return float(sum(_weighted_sum(df, column) for column in columns))


def _weighted_sum(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    values = (
        pd.to_numeric(df[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    weights = (
        pd.to_numeric(df["S006"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
        / 100.0
    )
    return _json_float(float(np.dot(values, weights)))


def _s006_sum(df: pd.DataFrame) -> float:
    if df.empty or "S006" not in df.columns:
        return 0.0
    total = (
        pd.to_numeric(df["S006"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .sum()
    )
    return _json_float(float(total))


def _s006_weight_total(df: pd.DataFrame) -> float:
    return _json_float(_s006_sum(df) / 100.0)


def _relative_error(estimate: float, target: float) -> float:
    denominator = max(abs(float(target)), 1.0)
    return float((float(estimate) - float(target)) / denominator)


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _json_float(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Audit value must be finite for JSON output: {value!r}.")
    return result


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


def _get_disaggregation_amount_columns(columns: pd.Index | list[str]) -> list[str]:
    amount_columns = [
        column
        for column in _get_amount_columns(columns)
        if column not in _DIVIDEND_RAW_AMOUNT_COLUMNS
    ]
    amount_columns.extend(
        column
        for column in _DIVIDEND_COMPONENT_AMOUNT_COLUMNS
        if column in columns and column not in amount_columns
    )
    return amount_columns


def _derive_default_puf_policyengine_variables_if_available(
    puf: pd.DataFrame,
) -> pd.DataFrame:
    sources = {
        _DEFAULT_PUF_POLICYENGINE_VARIABLES["ordinary_dividend_source"],
        _DEFAULT_PUF_POLICYENGINE_VARIABLES["qualified_dividend_source"],
    }
    if not sources.issubset(puf.columns):
        return puf.copy()
    return derive_puf_policyengine_variables(
        puf,
        **_DEFAULT_PUF_POLICYENGINE_VARIABLES,
    )


def _numeric_series(values: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def _assert_dividend_source_invariant(
    ordinary: pd.Series,
    qualified: pd.Series,
    *,
    ordinary_source: str,
    qualified_source: str,
) -> None:
    negative_ordinary = ordinary < -_DIVIDEND_INVARIANT_ATOL
    if bool(negative_ordinary.any()):
        value = float(ordinary.loc[negative_ordinary].min())
        raise ValueError(
            f"PUF source column {ordinary_source!r} contains a negative ordinary "
            f"dividend value ({value})."
        )
    negative_qualified = qualified < -_DIVIDEND_INVARIANT_ATOL
    if bool(negative_qualified.any()):
        value = float(qualified.loc[negative_qualified].min())
        raise ValueError(
            f"PUF source column {qualified_source!r} contains a negative qualified "
            f"dividend value ({value})."
        )
    above_ordinary = qualified > ordinary + _DIVIDEND_INVARIANT_ATOL
    if bool(above_ordinary.any()):
        gap = float((qualified - ordinary).loc[above_ordinary].max())
        raise ValueError(
            f"PUF source column {qualified_source!r} has qualified dividends above "
            f"ordinary dividends in {ordinary_source!r} by as much as {gap}."
        )


def _reconcile_puf_dividend_columns_from_components(puf: pd.DataFrame) -> pd.DataFrame:
    required = {"qualified_dividend_income", "non_qualified_dividend_income"}
    if not required.issubset(puf.columns):
        return _derive_default_puf_policyengine_variables_if_available(puf)

    result = puf.copy()
    qualified = _numeric_series(result["qualified_dividend_income"])
    non_qualified = _numeric_series(result["non_qualified_dividend_income"])
    _assert_nonnegative_dividend_component(
        qualified,
        column="qualified_dividend_income",
    )
    _assert_nonnegative_dividend_component(
        non_qualified,
        column="non_qualified_dividend_income",
    )
    ordinary = qualified + non_qualified
    result["qualified_dividend_income"] = qualified
    result["non_qualified_dividend_income"] = non_qualified
    if "E00650" in result.columns:
        result["E00650"] = qualified
    if "E00600" in result.columns:
        result["E00600"] = ordinary
    return result


def _reconcile_puf_qualified_tuition_from_sources(
    puf: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute tuition after aggregate-row source amounts are replaced.

    ``derive_puf_policyengine_variables`` runs before aggregate-record
    disaggregation.  The disaggregator subsequently reallocates the raw
    ``E03230``/``E87530`` amounts, so carrying the earlier derived column would
    leave donor-template values that no longer agree with either source field.
    Re-deriving here preserves the retired ``max(E03230, E87530)`` contract on
    every regular and synthetic row.
    """

    output = "qualified_tuition_expenses"
    primary = "E03230"
    optional = "E87530"
    if output not in puf.columns or primary not in puf.columns:
        return puf

    result = puf.copy()
    tuition = _numeric_series(result[primary]).clip(lower=0.0)
    if optional in result.columns:
        optional_tuition = _numeric_series(result[optional]).clip(lower=0.0)
        tuition = pd.Series(
            np.maximum(tuition.to_numpy(), optional_tuition.to_numpy()),
            index=result.index,
            dtype="float64",
        )
    result[output] = tuition
    return result


def _assert_nonnegative_dividend_component(
    values: pd.Series,
    *,
    column: str,
) -> None:
    negative = values < -_DIVIDEND_INVARIANT_ATOL
    if bool(negative.any()):
        value = float(values.loc[negative].min())
        raise ValueError(
            f"PUF dividend component {column!r} contains a negative value ({value})."
        )


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
