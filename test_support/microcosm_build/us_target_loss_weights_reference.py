"""The national target-loss weighting exactly as it stood before the move.

A verbatim copy of the five functions and two constants that
``tools/build_us_fiscal_refresh_release.py`` defined at 48fa06166 (origin/main
when the weighting moved to ``microcosm.build.us_runtime.target_loss_weights``),
under their original names, so ``git diff`` against that commit's tool shows
them unchanged. The differential tests hold the shared module to this copy.
Do not edit it; it is the "before" of a pure move.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from microcosm.calibrate import TargetRegistry

US_FISCAL_TARGET_VALUE_WEIGHT_POWER = 0.5

US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS = frozenset(
    {
        "congressional_district_geoid",
        "geography_scope",
        "hierarchy_child_ids",
        "hierarchy_child_sum_raw",
        "hierarchy_coverage_ratio",
        "hierarchy_expected_child_count",
        "hierarchy_observed_child_count",
        "hierarchy_parent_geography_id",
        "hierarchy_parent_geography_level",
        "hierarchy_parent_key",
        "hierarchy_parent_target_name",
        "hierarchy_parent_target_period",
        "hierarchy_parent_value",
        "hierarchy_raw_value",
        "hierarchy_reconciliation_factor",
        "hierarchy_reconciliation_method",
        "hierarchy_reconciliation_rule",
        "ledger_aggregate_fact_key",
        "ledger_dimension_set_key",
        "ledger_fact_key",
        "ledger_geography_id",
        "ledger_geography_level",
        "ledger_geography_name",
        "ledger_geography_vintage",
        "ledger_legacy_fact_key",
        "ledger_layout_groupby_dimension",
        "ledger_layout_groupby_value_id",
        "ledger_layout_record_set_id",
        "ledger_observed_measure_key",
        "ledger_semantic_fact_key",
        "ledger_source_record_id",
        "state_fips",
    }
)


def _fiscal_target_loss_weights(
    registry: TargetRegistry,
    family_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    weights = _fiscal_target_concept_budget_weights(registry)
    bases = np.asarray(
        [_fiscal_target_value_basis(spec) for spec in registry.specs],
        dtype=object,
    )
    unique_bases = sorted(set(bases.tolist()))
    if not unique_bases:
        return weights
    basis_total = len(weights) / len(unique_bases)
    for basis in unique_bases:
        mask = bases == basis
        current_total = weights[mask].sum()
        if current_total > 0:
            weights[mask] *= basis_total / current_total
    weights = weights / weights.mean()
    if not family_multipliers:
        return weights
    families = np.asarray(
        [spec.family for spec in registry.specs],
        dtype=object,
    )
    for family, multiplier in sorted(family_multipliers.items()):
        mask = families == family
        if not mask.any():
            raise ValueError(
                f"--target-family-loss-multiplier family {family!r} matches "
                "no compiled target."
            )
        weights[mask] *= multiplier
    return weights / weights.mean()


def _fiscal_target_concept_budget_weights(registry: TargetRegistry) -> np.ndarray:
    weights = _fiscal_target_value_basis_weights(registry)
    group_indices: dict[tuple[object, ...], list[int]] = {}
    for index, spec in enumerate(registry.specs):
        group_indices.setdefault(_fiscal_target_concept_budget_key(spec), []).append(
            index
        )
    for indices in group_indices.values():
        group_weights = weights[indices]
        group_total = float(group_weights.sum())
        group_budget = float(group_weights.max(initial=0.0))
        if group_total > 0 and group_budget > 0:
            weights[indices] *= group_budget / group_total
    return weights


def _fiscal_target_concept_budget_key(spec) -> tuple[object, ...]:
    metadata = spec.metadata
    if metadata.get("ledger_geography_level") != "congressional_district":
        return (
            _fiscal_target_value_basis(spec),
            spec.entity,
            spec.period,
            spec.family,
            spec.name,
        )
    semantic_metadata = tuple(
        sorted(
            (key, value)
            for key, value in metadata.items()
            if key not in US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
        )
    )
    return (
        _fiscal_target_value_basis(spec),
        spec.entity,
        spec.period,
        spec.family,
        spec.filter or "",
        metadata.get("state_fips", ""),
        semantic_metadata,
    )


def _fiscal_target_value_basis_weights(registry: TargetRegistry) -> np.ndarray:
    weights = np.ones(len(registry.specs), dtype=np.float64)
    bases = np.asarray(
        [_fiscal_target_value_basis(spec) for spec in registry.specs],
        dtype=object,
    )
    values = np.asarray(
        [max(abs(float(spec.value)), 1.0) for spec in registry.specs],
        dtype=np.float64,
    )
    raw_weights = values**US_FISCAL_TARGET_VALUE_WEIGHT_POWER
    for basis in sorted(set(bases.tolist())):
        mask = bases == basis
        mean_value = raw_weights[mask].mean()
        if mean_value > 0:
            weights[mask] = raw_weights[mask] / mean_value
    return weights


def _fiscal_target_value_basis(spec) -> str:
    metadata = spec.metadata
    measure_mode = metadata.get("measure_mode", "")
    source_measure_id = metadata.get("source_measure_id", "")
    if measure_mode in {
        "indicator_sum",
        "less_than_indicator_sum",
    }:
        return "count"
    if "enrollment" in source_measure_id or "recipients" in source_measure_id:
        return "count"
    if "return" in source_measure_id and "count" in source_measure_id:
        return "count"
    return "amount"
