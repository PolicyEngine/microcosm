"""Declared AGI own-tail donor eligibility and deterministic budget thinning."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.puf_support import (
    PUF_TAIL_PERSON_PROJECTION_ATTR,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_PROXY_AGI_COMPONENTS,
    puf_tail_person_projection,
)
from microcosm.build.us_runtime.qbi_inputs import US_QBI_BOOLEAN_OUTPUT_COLUMNS

# Declared source-stratum boundary, never a tuning knob: the measured pooled
# ASEC person-income ceiling is $3.15M, and only 18 of 231,007 QRF clones reach
# $5M. The processed PUF contains 19,034 donors at or above this boundary.
# A whole-dollar boundary, declared as an integer so the live identity and the
# canonical spec projection (which normalizes integral floats to integers)
# carry the same scalar.
PUF_AGI_TAIL_FLOOR = 5_000_000
PUF_AGI_TAIL_MAX_COUNT = 3_000


def puf_agi_tail_selection_identity() -> dict[str, object]:
    """Bind source eligibility, typed role transfer, and budget semantics."""

    return {
        "version": 1,
        "proxy_agi_components": list(PUF_TAX_DETAIL_PROXY_AGI_COMPONENTS),
        "proxy_agi_floor": PUF_AGI_TAIL_FLOOR,
        "floor_comparison": "greater_than_or_equal",
        "source_levels": "donor_frame_build_period",
        "maximum_agi_only_donors": PUF_AGI_TAIL_MAX_COUNT,
        "arms": {"1": "capital_gains", "2": "agi", "3": "both"},
        "eligibility": (
            "positive_weight_source_eligible; exactly_one_head_at_most_one_spouse; "
            "only_head_spouse_dependent_roles; finite_typed_owned_values; "
            "dependent_monetary_owned_values_exact_zero; "
            "person_monetary_sums_equal_tax_unit_donor"
        ),
        "person_transfer": "head_to_head_spouse_to_spouse_other_members_zero_false",
        "spouse_compatibility": "spouse_required_if_any_owned_spouse_value_nonzero",
        "ineligible_agi_candidate": "skip_entire_donor_including_both_arm",
        "thinning_cells": "filing_status_and_unweighted_proxy_agi_rank_decile",
        "decile_order": "proxy_agi_then_source_id_over_agi_only_arm",
        "quota": "one_per_cell_then_capacity_proportional_largest_remainder",
        "selection": "systematic_midpoints_in_source_id_order_no_rng",
        "weights": "proportional_rescale_per_cell_with_exact_float_sum_residual",
        "capital_gains_donors": "never_thinned",
        "legacy_input": "incomplete_proxy_surface_without_projection_is_cg_only",
    }


def _mass(rows: pd.DataFrame) -> dict[str, object]:
    weights = rows["weight"].to_numpy(dtype=np.float64)
    proxy = rows["_puf_tail_proxy_agi"].to_numpy(dtype=np.float64)
    return {
        "count": len(rows),
        "weight": float(weights.sum()),
        "proxy_agi_sum": float(proxy.sum()),
        "weighted_proxy_agi": float(np.dot(weights, proxy)),
    }


def _aligned_mask(mask: np.ndarray, donor: pd.DataFrame, label: str) -> np.ndarray:
    values = np.asarray(mask)
    if values.dtype.kind != "b" or values.shape != (len(donor),):
        raise ValueError(f"AGI tail {label} must be an aligned boolean mask.")
    return values


def _person_vectors(
    donor: pd.Series,
    persons: pd.DataFrame,
    owned: tuple[str, ...],
) -> tuple[str | None, dict[str, object]]:
    roles = persons["role"].astype(str)
    if (
        roles.eq("head").sum() != 1
        or roles.eq("spouse").sum() > 1
        or not roles.isin(("head", "spouse", "dependent")).all()
    ):
        return "unrepresentable_person_roles", {}
    bool_columns = set(US_QBI_BOOLEAN_OUTPUT_COLUMNS)
    monetary = tuple(column for column in owned if column not in bool_columns)
    for column in owned:
        if column not in persons:
            return "missing_person_output", {}
        dtype = persons[column].dtype
        if column in bool_columns:
            if dtype != np.dtype(bool):
                return "unsupported_person_dtype", {}
        elif dtype.kind not in "iuf":
            return "unsupported_person_dtype", {}
        if not np.isfinite(persons[column].to_numpy()).all():
            return "nonfinite_person_output", {}
    dependents = persons.loc[roles.eq("dependent"), list(monetary)]
    if (dependents.to_numpy() != 0).any():
        return "dependent_monetary_value", {}
    if any(float(persons[column].sum()) != float(donor[column]) for column in monetary):
        return "person_tax_unit_monetary_mismatch", {}
    vectors: dict[str, dict[str, object]] = {}
    source_ids: dict[str, int] = {}
    for role in ("head", "spouse"):
        role_rows = persons.loc[roles.eq(role)]
        if role_rows.empty:
            continue
        values = role_rows.iloc[0]
        source_ids[role] = int(values["person_id"])
        vectors[role] = {
            column: bool(values[column])
            if column in bool_columns
            else values[column].item()
            if isinstance(values[column], np.generic)
            else values[column]
            for column in owned
        }
    return None, {
        "vectors": vectors,
        "dtypes": {column: str(persons[column].dtype) for column in owned},
        "source_ids": source_ids,
        "needs_spouse": any(value != 0 for value in vectors.get("spouse", {}).values()),
    }


def _cell_quotas(counts: np.ndarray) -> np.ndarray:
    budget = min(PUF_AGI_TAIL_MAX_COUNT, int(counts.sum()))
    if len(counts) > budget:
        raise ValueError("AGI tail record budget cannot preserve every thinning cell.")
    if budget == int(counts.sum()):
        return counts.copy()
    quotas = np.ones(len(counts), dtype=np.int64)
    capacity = counts - quotas
    remaining = budget - len(counts)
    ideal = capacity * (remaining / float(capacity.sum()))
    extra = np.floor(ideal).astype(np.int64)
    quotas += extra
    missing = budget - int(quotas.sum())
    order = np.argsort(-(ideal - extra), kind="stable")
    quotas[order[:missing]] += 1
    return quotas


def _rescale_exact(weights: np.ndarray, total: float) -> np.ndarray:
    result = weights * (total / float(weights.sum()))
    for _attempt in range(8):
        observed = float(result.sum())
        if observed == total:
            return result
        result[-1] += total - observed
    for _attempt in range(64):
        observed = float(result.sum())
        if observed == total:
            return result
        result[-1] = np.nextafter(result[-1], np.inf if observed < total else -np.inf)
    raise ValueError("AGI tail thinning cannot exactly represent cell weight mass.")


def _thin_agi_only(agi_only: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    ranked = agi_only.sort_values(
        ["_puf_tail_proxy_agi", "tax_unit_id"], kind="stable"
    ).copy()
    ranked["_decile"] = (
        np.arange(len(ranked), dtype=np.int64) * 10 // max(len(ranked), 1)
    )
    cells = list(ranked.groupby(["filing_status_code", "_decile"], sort=True))
    quotas = _cell_quotas(
        np.asarray([len(cell) for _key, cell in cells], dtype=np.int64)
    )
    kept_cells = []
    receipts = []
    for ((filing_status, decile), cell), quota in zip(cells, quotas, strict=True):
        ordered = cell.sort_values("tax_unit_id", kind="stable").copy()
        before = _mass(ordered)
        if quota == len(ordered):
            kept = ordered.copy()
        else:
            indices = np.floor((np.arange(quota) + 0.5) * len(ordered) / quota).astype(
                int
            )
            kept = ordered.iloc[indices].copy()
            kept["weight"] = _rescale_exact(
                kept["weight"].to_numpy(dtype=np.float64), float(before["weight"])
            )
        after = _mass(kept)
        receipts.append(
            {
                "filing_status_code": float(filing_status),
                "proxy_agi_decile": int(decile),
                "count_before": len(ordered),
                "kept_count": len(kept),
                "dropped_count": len(ordered) - len(kept),
                "weight_before": before["weight"],
                "weight_after": after["weight"],
                "proxy_agi_sum_before": before["proxy_agi_sum"],
                "proxy_agi_sum_after": after["proxy_agi_sum"],
                "weighted_proxy_agi_before": before["weighted_proxy_agi"],
                "weighted_proxy_agi_after": after["weighted_proxy_agi"],
                "kept_donor_source_ids": kept["tax_unit_id"].astype(int).tolist(),
            }
        )
        kept_cells.append(kept.drop(columns="_decile"))
    kept = (
        pd.concat(kept_cells, ignore_index=True)
        if kept_cells
        else agi_only.iloc[:0].copy()
    )
    return kept, {
        "maximum_count": PUF_AGI_TAIL_MAX_COUNT,
        "count_before": len(agi_only),
        "kept_count": len(kept),
        "dropped_count": len(agi_only) - len(kept),
        "cells": receipts,
    }


def select_puf_agi_tail_donors(
    donor: pd.DataFrame,
    capital_gains_mask: np.ndarray,
    eligible: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Union gains and AGI donors, failing closed on unrepresentable AGI rows."""

    from microcosm.build.us_runtime.puf_capital_gains_tail import puf_tail_owned_columns

    gains = _aligned_mask(capital_gains_mask, donor, "capital-gains mask")
    source_eligible = _aligned_mask(eligible, donor, "source eligibility")
    required = {"tax_unit_id", "weight", "filing_status_code"}
    if required - set(donor):
        raise ValueError(
            f"AGI tail donor is missing columns: {sorted(required - set(donor))}."
        )
    if donor.tax_unit_id.duplicated().any():
        raise ValueError("AGI tail donor source IDs must be unique.")
    for column in required:
        if not np.isfinite(pd.to_numeric(donor[column], errors="raise")).all():
            raise ValueError(f"AGI tail donor {column} must be finite.")
    if donor.weight.lt(0).any():
        raise ValueError("AGI tail donor weights must be nonnegative.")
    source_ids = donor.tax_unit_id.to_numpy(dtype=np.float64)
    if not np.equal(source_ids, np.floor(source_ids)).all():
        raise ValueError("AGI tail donor source IDs must be integers.")
    result = donor.copy()
    result.attrs = {}
    full_surface = {
        *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    }
    complete = full_surface <= set(donor)
    proxy_complete = set(PUF_TAX_DETAIL_PROXY_AGI_COMPONENTS) <= set(donor)
    has_projection = PUF_TAIL_PERSON_PROJECTION_ATTR in donor.attrs
    if not complete and has_projection:
        raise ValueError(
            "AGI tail person projection requires the complete donor output vector."
        )
    proxy = np.zeros(len(donor), dtype=np.float64)
    if proxy_complete:
        for column in PUF_TAX_DETAIL_PROXY_AGI_COMPONENTS:
            values = pd.to_numeric(donor[column], errors="raise").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(f"AGI tail proxy component {column!r} must be finite.")
            proxy += values
    agi = (
        source_eligible & (proxy >= PUF_AGI_TAIL_FLOOR) & donor.weight.gt(0).to_numpy()
    )
    if agi.any() and not complete:
        raise ValueError(
            "AGI tail donors require the complete donor output vector; "
            f"missing columns: {sorted(full_surface - set(donor))}."
        )
    if agi.any() and not has_projection:
        raise ValueError("AGI tail donors require a bound person projection.")
    result["_puf_tail_proxy_agi"] = proxy
    result["_puf_tail_arm"] = gains.astype(np.int8) + 2 * agi.astype(np.int8)
    for column in (
        "_puf_tail_person_vectors",
        "_puf_tail_person_dtypes",
        "_puf_tail_person_source_ids",
        "_puf_tail_tax_unit_dtypes",
    ):
        result[column] = pd.Series(
            [{} for _ in range(len(result))], index=result.index, dtype=object
        )
    result["_puf_tail_needs_spouse"] = False
    skipped: dict[str, list[int]] = {}
    if agi.any():
        persons = puf_tail_person_projection(donor)
        owned_surface = puf_tail_owned_columns(2)
        owned = owned_surface["person"]
        tax_unit_dtypes = {
            column: str(donor[column].dtype) for column in owned_surface["tax_unit"]
        }
        grouped = {
            int(key): group for key, group in persons.groupby("tax_unit_id", sort=False)
        }
        for position in np.flatnonzero(agi):
            row = result.iloc[position]
            group = grouped.get(int(row.tax_unit_id), persons.iloc[:0])
            reason = None
            if any(donor[column].dtype.kind not in "iuf" for column in tax_unit_dtypes):
                reason = "unsupported_tax_unit_dtype"
            elif not np.isfinite(
                row[list(tax_unit_dtypes)].to_numpy(dtype=float)
            ).all():
                reason = "nonfinite_tax_unit_output"
            payload = {}
            if reason is None:
                reason, payload = _person_vectors(row, group, owned)
            if reason is not None:
                skipped.setdefault(reason, []).append(int(position))
                continue
            assert isinstance(payload, Mapping)
            for suffix, key in (
                ("vectors", "vectors"),
                ("dtypes", "dtypes"),
                ("source_ids", "source_ids"),
            ):
                result.iat[
                    position, result.columns.get_loc(f"_puf_tail_person_{suffix}")
                ] = payload[key]
            result.iat[position, result.columns.get_loc("_puf_tail_needs_spouse")] = (
                payload["needs_spouse"]
            )
            result.iat[
                position, result.columns.get_loc("_puf_tail_tax_unit_dtypes")
            ] = tax_unit_dtypes.copy()
    skipped_positions = sorted(
        position for positions in skipped.values() for position in positions
    )
    skipped_rows = result.iloc[skipped_positions].sort_values(
        "tax_unit_id", kind="stable"
    )
    keep = (gains | agi).copy()
    keep[skipped_positions] = False
    candidates = result.iloc[np.flatnonzero(keep)]
    unthinned = candidates.loc[candidates._puf_tail_arm.ne(2)]
    thinned, thinning = _thin_agi_only(candidates.loc[candidates._puf_tail_arm.eq(2)])
    selected = (
        pd.concat([unthinned, thinned], ignore_index=True)
        .sort_values("tax_unit_id", kind="stable")
        .reset_index(drop=True)
    )
    receipt = {
        "identity": puf_agi_tail_selection_identity(),
        "projection_status": "available"
        if has_projection
        else "not_required_no_agi_candidates"
        if proxy_complete
        else "unavailable_legacy_capital_gains_only",
        "agi_candidate_count": int(agi.sum()),
        "agi_selected_count": int(selected._puf_tail_arm.isin((2, 3)).sum()),
        "both_selected_count": int(selected._puf_tail_arm.eq(3).sum()),
        "skipped": {
            **_mass(skipped_rows),
            "donor_source_ids": skipped_rows.tax_unit_id.astype(int).tolist(),
            "by_reason": {
                reason: _mass(
                    result.iloc[positions].sort_values("tax_unit_id", kind="stable")
                )
                for reason, positions in sorted(skipped.items())
            },
        },
        "thinning": thinning,
    }
    validate_puf_agi_tail_selection_receipt(receipt)
    return selected, receipt


def validate_puf_agi_tail_selection_receipt(receipt: Mapping[str, object]) -> None:
    """Validate exact selection semantics and independent mass/count equations."""

    def require_keys(value: object, keys: set[str], label: str) -> Mapping:
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(f"AGI tail {label} has an invalid schema.")
        return value

    def count(value: object, label: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"AGI tail {label} must be a nonnegative integer.")
        return value

    def amount(value: object, label: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
            raise ValueError(f"AGI tail {label} must be a finite nonnegative amount.")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"AGI tail {label} must be a finite nonnegative amount.")
        return float(value)

    def source_ids(value: object, expected_count: int, label: str) -> list[int]:
        if (
            not isinstance(value, list)
            or len(value) != expected_count
            or any(type(item) is not int for item in value)
            or len(set(value)) != expected_count
            or value != sorted(value)
        ):
            raise ValueError(
                f"AGI tail {label} must contain unique ordered source IDs."
            )
        return value

    mass_keys = {"count", "weight", "proxy_agi_sum", "weighted_proxy_agi"}

    def mass(value: object, label: str, extra_keys: set[str] | None = None) -> Mapping:
        validated = require_keys(value, mass_keys | (extra_keys or set()), label)
        n = count(validated["count"], f"{label} count")
        for name in mass_keys - {"count"}:
            magnitude = amount(validated[name], f"{label} {name}")
            if (n == 0) != (magnitude == 0):
                raise ValueError(f"AGI tail {label} count and {name} disagree.")
        return validated

    validated = require_keys(
        receipt,
        {
            "identity",
            "projection_status",
            "agi_candidate_count",
            "agi_selected_count",
            "both_selected_count",
            "skipped",
            "thinning",
        },
        "selection receipt",
    )
    try:
        observed_identity = json.dumps(
            validated["identity"], sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "AGI tail selection identity is not canonical JSON."
        ) from error
    if observed_identity != json.dumps(
        puf_agi_tail_selection_identity(), sort_keys=True, allow_nan=False
    ):
        raise ValueError(
            "AGI tail selection identity differs from the declared contract."
        )
    statuses = {
        "available",
        "not_required_no_agi_candidates",
        "unavailable_legacy_capital_gains_only",
    }
    if (
        not isinstance(validated["projection_status"], str)
        or validated["projection_status"] not in statuses
    ):
        raise ValueError("AGI tail selection has an unknown projection status.")
    candidates = count(validated["agi_candidate_count"], "candidate count")
    selected = count(validated["agi_selected_count"], "selected count")
    both = count(validated["both_selected_count"], "both-arm count")
    skipped = mass(
        validated["skipped"], "skipped donors", {"donor_source_ids", "by_reason"}
    )
    skipped_ids = source_ids(
        skipped["donor_source_ids"], skipped["count"], "skipped donors"
    )
    reasons = skipped["by_reason"]
    allowed_reasons = {
        "unrepresentable_person_roles",
        "missing_person_output",
        "unsupported_person_dtype",
        "nonfinite_person_output",
        "dependent_monetary_value",
        "person_tax_unit_monetary_mismatch",
        "unsupported_tax_unit_dtype",
        "nonfinite_tax_unit_output",
    }
    if not isinstance(reasons, Mapping) or set(reasons) - allowed_reasons:
        raise ValueError("AGI tail skipped donors have an unknown eligibility reason.")
    reason_masses = [
        mass(value, f"skip reason {name}") for name, value in reasons.items()
    ]
    if sum(item["count"] for item in reason_masses) != skipped["count"]:
        raise ValueError("AGI tail skip reason counts do not sum to skipped donors.")
    for name in mass_keys - {"count"}:
        if not np.isclose(
            sum(item[name] for item in reason_masses), skipped[name], rtol=1e-14, atol=0
        ):
            raise ValueError(
                f"AGI tail skip reason {name} does not conserve reported mass."
            )
    thinning = require_keys(
        validated["thinning"],
        {"maximum_count", "count_before", "kept_count", "dropped_count", "cells"},
        "thinning receipt",
    )
    if count(thinning["maximum_count"], "maximum count") != PUF_AGI_TAIL_MAX_COUNT:
        raise ValueError("AGI tail thinning maximum differs from the declared budget.")
    before = count(thinning["count_before"], "thinning input count")
    kept = count(thinning["kept_count"], "thinning kept count")
    dropped = count(thinning["dropped_count"], "thinning dropped count")
    if before != kept + dropped or kept != min(before, PUF_AGI_TAIL_MAX_COUNT):
        raise ValueError(
            "AGI tail thinning counts do not conserve the declared budget."
        )
    cells = thinning["cells"]
    if not isinstance(cells, list):
        raise ValueError("AGI tail thinning cells must be a list.")
    cell_keys = {
        "filing_status_code",
        "proxy_agi_decile",
        "count_before",
        "kept_count",
        "dropped_count",
        "weight_before",
        "weight_after",
        "proxy_agi_sum_before",
        "proxy_agi_sum_after",
        "weighted_proxy_agi_before",
        "weighted_proxy_agi_after",
        "kept_donor_source_ids",
    }
    cell_inputs, cell_kept, ordered_keys, all_ids = [], [], [], []
    for cell in cells:
        cell = require_keys(cell, cell_keys, "thinning cell")
        status = amount(cell["filing_status_code"], "cell filing status")
        decile = count(cell["proxy_agi_decile"], "cell decile")
        n = count(cell["count_before"], "cell input count")
        k = count(cell["kept_count"], "cell kept count")
        d = count(cell["dropped_count"], "cell dropped count")
        if decile > 9 or k == 0 or n != k + d:
            raise ValueError("AGI tail thinning cell has invalid counts or decile.")
        for name in (
            "weight_before",
            "weight_after",
            "proxy_agi_sum_before",
            "proxy_agi_sum_after",
            "weighted_proxy_agi_before",
            "weighted_proxy_agi_after",
        ):
            if amount(cell[name], f"cell {name}") == 0:
                raise ValueError("AGI tail nonempty thinning cell has zero mass.")
        if cell["weight_before"] != cell["weight_after"]:
            raise ValueError(
                "AGI tail thinning cell weight mass is not exactly conserved."
            )
        if d == 0 and any(
            cell[f"{name}_before"] != cell[f"{name}_after"]
            for name in ("proxy_agi_sum", "weighted_proxy_agi")
        ):
            raise ValueError("AGI tail unchanged thinning cell proxy mass changed.")
        all_ids.extend(source_ids(cell["kept_donor_source_ids"], k, "thinning cell"))
        ordered_keys.append((status, decile))
        cell_inputs.append(n)
        cell_kept.append(k)
    if ordered_keys != sorted(set(ordered_keys)):
        raise ValueError("AGI tail thinning cell keys must be unique and ordered.")
    if sum(cell_inputs) != before or sum(cell_kept) != kept:
        raise ValueError("AGI tail thinning cells do not sum to the receipt counts.")
    if cell_kept != _cell_quotas(np.asarray(cell_inputs, dtype=np.int64)).tolist():
        raise ValueError(
            "AGI tail thinning cell quotas differ from the declared allocation."
        )
    if len(set(all_ids)) != len(all_ids) or set(all_ids) & set(skipped_ids):
        raise ValueError(
            "AGI tail donor source IDs overlap across cells or skipped donors."
        )
    if candidates != selected + skipped["count"] + dropped or selected != both + kept:
        raise ValueError("AGI tail selected/skipped/thinned count equation failed.")
    if validated["projection_status"] != "available" and candidates:
        raise ValueError("AGI tail candidates require an available person projection.")


def validate_puf_tail_vector_mass_receipts(manifest: Mapping[str, object]) -> None:
    """Recompute every transferred-column receipt from the bound donor records."""

    from microcosm.build.us_runtime.puf_capital_gains_tail import puf_tail_owned_columns

    def numeric(value: object, label: str, *, boolean: bool = False) -> float:
        if (
            not isinstance(value, (int, float))
            or (isinstance(value, bool) and not boolean)
            or not np.isfinite(value)
        ):
            raise ValueError(f"PUF tail {label} must be a finite numeric value.")
        return float(value)

    def mapping(value: object, label: str, keys: set[str] | None = None) -> Mapping:
        if not isinstance(value, Mapping) or (keys is not None and set(value) != keys):
            raise ValueError(f"PUF tail {label} has an invalid receipt surface.")
        return value

    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("PUF tail vector mass validation requires donor records.")
    domain = mapping(manifest.get("weight_domain"), "weight domain")
    normalization = numeric(domain.get("design_weight_normalization"), "normalization")
    if normalization <= 0:
        raise ValueError("PUF tail weight normalization must be positive.")
    cg_owned = puf_tail_owned_columns(1)
    agi_owned = puf_tail_owned_columns(2)
    cg_columns = (*cg_owned["person"], *cg_owned["tax_unit"])
    full_columns = (*agi_owned["person"], *agi_owned["tax_unit"])
    additional = set(full_columns) - set(cg_columns)
    expected_full: dict[str, dict[str, object]] = {}
    source_ids, donor_weights, assigned_weights = [], [], []
    joint = {column: [] for column in cg_columns}
    checked_cells = 0
    for raw_record in records:
        record = mapping(raw_record, "donor record")
        arm = record.get("arm")
        try:
            puf_tail_owned_columns(arm)
        except ValueError as error:
            raise ValueError(
                "PUF tail vector receipt has unknown arm provenance."
            ) from error
        source_id = record.get("donor_source_id")
        if type(source_id) is not int:
            raise ValueError("PUF tail vector donor source IDs must be integers.")
        source_ids.append(source_id)
        donor_weight = numeric(record.get("donor_weight"), "donor weight")
        assigned_weight = numeric(record.get("assigned_weight"), "assigned weight")
        if donor_weight <= 0 or assigned_weight != donor_weight * normalization:
            raise ValueError(
                "PUF tail donor/assigned weights do not match normalization."
            )
        donor_weights.append(donor_weight)
        assigned_weights.append(assigned_weight)
        count = record.get("tail_person_count")
        if type(count) is not int or count <= 0:
            raise ValueError("PUF tail person count must be a positive integer.")
        vector = mapping(record.get("joint_vector"), "joint vector", set(cg_columns))
        for column in cg_columns:
            joint[column].append(numeric(vector[column], f"joint vector {column}"))
        if arm == 1:
            continue
        roles = mapping(record.get("person_vectors"), "person vectors")
        if "head" not in roles or set(roles) - {"head", "spouse"}:
            raise ValueError("PUF tail person count/roles cannot represent the vector.")
        for role, role_vector in roles.items():
            mapping(role_vector, f"{role} vector", set(agi_owned["person"]))
        # Keep a zero-valued source spouse in donor provenance even when the
        # recipient has only a head. The eligibility rule requires an actual
        # spouse carrier only for a nonzero owned source-spouse value.
        required_people = 1 + int(any(roles.get("spouse", {}).values()))
        if count < required_people:
            raise ValueError("PUF tail person count/roles cannot represent the vector.")
        unit = mapping(
            record.get("tax_unit_vector"), "tax-unit vector", set(agi_owned["tax_unit"])
        )
        totals = {
            column: sum(
                numeric(role[column], f"person vector {column}", boolean=True)
                for role in roles.values()
            )
            for column in agi_owned["person"]
        }
        totals.update(
            {
                column: numeric(unit[column], f"tax-unit vector {column}")
                for column in agi_owned["tax_unit"]
            }
        )
        for column in cg_columns:
            if totals[column] != vector[column]:
                raise ValueError(
                    f"PUF tail AGI and joint vectors disagree for {column}."
                )
        checked_cells += len(agi_owned["person"]) * count + len(agi_owned["tax_unit"])
        for column in sorted(additional):
            receipt = expected_full.setdefault(
                column,
                {
                    "scope": "agi_arm",
                    "donor_weighted_signed_mass": 0.0,
                    "expected_frame_weighted_signed_mass": 0.0,
                    "transferred_frame_weighted_signed_mass": 0.0,
                    "difference": 0.0,
                },
            )
            receipt["donor_weighted_signed_mass"] += totals[column] * donor_weight
            receipt["expected_frame_weighted_signed_mass"] += (
                totals[column] * assigned_weight
            )
            receipt["transferred_frame_weighted_signed_mass"] += (
                totals[column] * assigned_weight
            )
    if source_ids != sorted(set(source_ids)):
        raise ValueError("PUF tail vector donor source IDs must be unique and ordered.")
    full = mapping(
        manifest.get("full_vector_reconciliation"),
        "full-vector reconciliation",
        {"passed", "owned_cell_count", "signed_mass"},
    )
    if full["passed"] is not True:
        raise ValueError("PUF tail full-vector reconciliation must pass.")
    if (
        type(full["owned_cell_count"]) is not int
        or full["owned_cell_count"] != checked_cells
    ):
        raise ValueError(
            "PUF tail full-vector checked cell count differs from records."
        )
    expected_signed = dict(expected_full)
    for column in cg_columns:
        donor_mass = float(np.dot(joint[column], donor_weights))
        expected = donor_mass * normalization
        transferred = float(np.dot(joint[column], assigned_weights))
        # Preserve the reviewed CG normalization arithmetic tolerance exactly.
        if not np.isclose(transferred, expected, rtol=1e-12, atol=1e-6):
            raise ValueError(
                f"PUF tail joint-vector signed mass is not conserved for {column}."
            )
        expected_signed[column] = {
            "donor_weighted_signed_mass": donor_mass,
            "design_weight_normalization": normalization,
            "expected_frame_weighted_signed_mass": expected,
            "transferred_frame_weighted_signed_mass": transferred,
            "difference": transferred - expected,
        }

    def compare(observed: object, expected: Mapping, label: str) -> None:
        columns = mapping(observed, label, set(expected))
        for column, expected_receipt in expected.items():
            row = mapping(columns[column], f"{label}/{column}", set(expected_receipt))
            for name, value in expected_receipt.items():
                if name == "scope":
                    if row[name] != value:
                        raise ValueError(
                            f"PUF tail {label}/{column} has the wrong arm scope."
                        )
                elif numeric(row[name], f"{label}/{column}/{name}") != value:
                    raise ValueError(
                        f"PUF tail {label}/{column}/{name} differs from donor records."
                    )

    compare(full["signed_mass"], expected_full, "full-vector signed mass")
    compare(
        manifest.get("signed_leg_reconciliation"),
        expected_signed,
        "signed-leg reconciliation",
    )
