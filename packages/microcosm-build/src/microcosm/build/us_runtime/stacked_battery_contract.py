"""JSON-shaped view of the live generation-0 stacked battery authority.

The F0 constants adapter needs an oracle that is independent of the compiled
country bundle.  This module reads the constants and public constructors used
by the live terminal gates, but never executes a gate or imports the spec
compiler.  The one-shot migration generator delegates here as well so there is
only one extraction of the generation-0 battery contract.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping

__all__ = ["build_live_stacked_battery_contract"]


_KERNEL_ID_BY_IMPLEMENTATION = {
    "microcosm.build.us_runtime.stacked_spine.by_origin_battery": ("by_origin_battery"),
    "microcosm.build.us_runtime.stacked_spine.stacked_completeness_gate": (
        "stacked_completeness_gate"
    ),
}


def _mapping_item_key(item: tuple[object, object]) -> str:
    """Return a deterministic sort key without a dynamic subscript."""

    key, _value = item
    return str(key)


def _json_ready(value: object) -> object:
    """Copy a live authority value into deterministic JSON-shaped data."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(nested)
            for key, nested in sorted(value.items(), key=_mapping_item_key)
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(nested) for nested in value]
    if isinstance(value, (set, frozenset)):
        return [_json_ready(nested) for nested in sorted(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _json_ready(item())
    raise TypeError(
        f"Live battery projections must be JSON-shaped; got {type(value).__name__}."
    )


def _kernel_ref(function: Callable[..., object]) -> str:
    implementation_id = f"{function.__module__}.{function.__qualname__}"
    try:
        kernel_id = _KERNEL_ID_BY_IMPLEMENTATION[implementation_id]
    except KeyError as error:  # pragma: no cover - closed live registry
        raise RuntimeError(
            f"No reviewed F0 battery kernel id for {implementation_id!r}."
        ) from error
    return f"kernel:{kernel_id}"


def build_live_stacked_battery_contract() -> dict[str, object]:
    """Extract the complete live stacked registry, thresholds, and gates."""

    stacked = importlib.import_module("microcosm.build.us_runtime.stacked_spine")
    metric_by_key = stacked.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    metric_by_physical_target: dict[tuple[str, str, str], tuple[int, str]] = {}
    for (entity, family, column, clone_index), metric in metric_by_key.items():
        physical_target = (entity, family, column)
        existing = metric_by_physical_target.get(physical_target)
        if existing is not None:
            raise RuntimeError(
                "Canonical battery metric registry has duplicate physical target "
                f"{physical_target!r} at clone roles {existing[0]!r} and "
                f"{clone_index!r}."
            )
        metric_by_physical_target[physical_target] = (clone_index, metric)

    registry = []
    ordered_physical_targets: set[tuple[str, str, str]] = set()
    for entity, families in stacked.CANONICAL_STACKED_DECLARED_SURFACE.items():
        for family, columns in families.items():
            for column in columns:
                physical_target = (entity, family, column)
                if physical_target in ordered_physical_targets:
                    raise RuntimeError(
                        "Canonical battery declared surface repeats physical target "
                        f"{physical_target!r}."
                    )
                try:
                    clone_index, metric = metric_by_physical_target[physical_target]
                except KeyError as error:
                    raise RuntimeError(
                        "Canonical battery surface has no metric registry row for "
                        f"{physical_target!r}."
                    ) from error
                registry.append(
                    {
                        "entity": entity,
                        "family": family,
                        "column": column,
                        "clone_index": clone_index,
                        "metric": metric,
                    }
                )
                ordered_physical_targets.add(physical_target)
    if ordered_physical_targets != set(metric_by_physical_target):
        raise RuntimeError(
            "Canonical battery metric registry and declared surface differ: "
            "missing="
            f"{sorted(set(metric_by_physical_target) - ordered_physical_targets)!r}, "
            "extra="
            f"{sorted(ordered_physical_targets - set(metric_by_physical_target))!r}."
        )
    joint_registry = [
        {
            "entity": entity,
            "family": family,
            "columns": list(columns),
            "clone_index": clone_index,
            "metric": metric,
        }
        for (entity, family, columns, clone_index), metric in sorted(
            stacked.CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY.items()
        )
    ]
    profile = stacked.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE
    authority_value = _json_ready(stacked.stacked_spine_authority_receipt())
    if not isinstance(authority_value, dict):  # pragma: no cover - live invariant
        raise RuntimeError("Stacked authority receipt must be a JSON object.")
    authority_receipt = authority_value
    thresholds = {
        "incidence_ratio": {
            "lower_inclusive": float(stacked._BATTERY_INCIDENCE_RATIO_BOUNDS[0]),
            "upper_inclusive": float(stacked._BATTERY_INCIDENCE_RATIO_BOUNDS[1]),
            "direction": "acs_over_asec",
            "both_zero": "fail_dead_comparison",
            "left_zero_right_positive": "positive_infinity_and_fail",
        },
        "conditional_quantile_envelope": {
            "probabilities": list(stacked._BATTERY_QUANTILES),
            "side": "left",
            "distance": "max(2*abs(asec-acs)/(abs(asec)+abs(acs)))",
            "maximum_inclusive": float(stacked._BATTERY_QUANTILE_ENVELOPE_TOLERANCE),
            "applies_separately_to": ["positive_magnitude", "negative_magnitude"],
        },
        "categorical_total_variation": {
            "distance": "0.5*sum(abs(asec_share-acs_share))",
            "maximum_inclusive": float(stacked._BATTERY_CATEGORICAL_TVD_TOLERANCE),
        },
    }
    common_gate_contract = {
        "input": {
            "artifact": "simulated_evaluation_frame",
            "stage": "terminal_gates",
        },
        "reference": {
            "kind": "compiled_authority",
            "authority_ref": "authority_binding",
        },
        "uncertainty": {"rule": "none"},
        "status": "fail",
    }
    return {
        "contract_id": "us_stacked_origin_battery",
        "version": int(profile.version),
        "gates": [
            {
                "id": stacked._COMPLETENESS_GATE_NAME,
                "kernel": _kernel_ref(stacked.stacked_completeness_gate),
                "metric": {
                    "formula_id": "declared_surface_completeness",
                    "params": {
                        "formula": (
                            "every_declared_positive_weight_cell_is_valid_and_"
                            "non_null_or_has_exact_absence_proof"
                        ),
                        "missing_or_null_in_scope": "fail",
                        "declared_structural_absence": ("requires_exact_absence_proof"),
                    },
                },
                **common_gate_contract,
                "population": {
                    "universe": "declared_positive_weight_rows",
                    "denominator": "declared_target_cells",
                },
                "slices": [
                    "entity",
                    "family",
                    "column",
                    "support_channel",
                    "support_clone_index",
                ],
                "min_support": 0,
                "effective_sample_size": {
                    "method": "not_computed",
                    "minimum": None,
                    "not_applicable_reason": (
                        "completeness_is_an_exact_cell_contract_not_an_estimate"
                    ),
                },
                "thresholds": {"absolute": 0.0, "relative": 0.0},
                "missing_slice": "fail",
                "reason_map": {
                    "authority_invalid": "authority_invalid",
                    "invalid_values": "invalid_declared_metric_values",
                    "missing": "missing_declared_target",
                    "missing_entity": "missing_declared_entity",
                    "structural_absence_mismatch": (
                        "structural_absence_equation_failed"
                    ),
                    "unproven": "unproven_declared_absence",
                },
            },
            {
                "id": stacked._BATTERY_GATE_NAME,
                "kernel": _kernel_ref(stacked.by_origin_battery),
                "metric": {
                    "formula_id": "by_origin_registry_battery",
                    "params": {
                        "formula": (
                            "registry_metric_by_asec_versus_acs_origin_on_"
                            "positive_weight_declared_clone_scope"
                        ),
                        "metric_contract_refs": list(
                            stacked.ORIGIN_BATTERY_METRIC_KINDS
                        ),
                        "missing_or_null_in_scope": "fail",
                        "declared_structural_absence": (
                            "exclude_only_after_exact_equation_proof"
                        ),
                    },
                },
                **common_gate_contract,
                "population": {
                    "universe": "positive_weight_declared_clone_rows",
                    "denominator": "resolved_entity_weight",
                },
                "slices": [
                    "entity",
                    "family",
                    "column",
                    "support_channel",
                    "support_clone_index",
                    "sign_leg",
                ],
                "min_support": int(profile.min_effective_support),
                "effective_sample_size": {
                    "method": "not_computed",
                    "minimum": None,
                    "not_applicable_reason": (
                        "current_authority_uses_unweighted_positive_weight_row_"
                        "count_per_origin"
                    ),
                },
                "thresholds": {
                    "absolute": max(
                        float(stacked._BATTERY_QUANTILE_ENVELOPE_TOLERANCE),
                        float(stacked._BATTERY_CATEGORICAL_TVD_TOLERANCE),
                    ),
                    "relative": max(
                        1.0 - float(stacked._BATTERY_INCIDENCE_RATIO_BOUNDS[0]),
                        float(stacked._BATTERY_INCIDENCE_RATIO_BOUNDS[1]) - 1.0,
                    ),
                },
                "missing_slice": "skip_with_receipt",
                "reason_map": {
                    "dead_comparison": "zero_incidence_both_origins",
                    "insufficient_support": "insufficient_origin_row_support",
                    "invalid_values": "invalid_declared_metric_values",
                    "missing_column": "missing_registered_column",
                    "null_in_scope": "null_inside_comparison_scope",
                    "structural_absence_mismatch": (
                        "structural_absence_equation_failed"
                    ),
                    "threshold_exceeded": "metric_threshold_exceeded",
                },
            },
        ],
        "comparison_scope": {
            "left_support_channel": stacked.BASE_ASEC_SUPPORT_CHANNEL,
            "right_support_channel": stacked.ACS_STACKED_SUPPORT_CHANNEL,
            "clone_index": "declared_per_registry_row",
            "weight_scope": "resolved_entity_weight > 0",
            "nulls": "fail_inside_scope",
            "invalid_values": "fail_inside_scope",
        },
        "support_profile": {
            "profile_id": str(profile.profile_id),
            "version": int(profile.version),
            "min_effective_support": int(profile.min_effective_support),
            "insufficient_support": "record_untestable_not_failure",
        },
        "metric_kinds": list(stacked.ORIGIN_BATTERY_METRIC_KINDS),
        "metric_registry": registry,
        "joint_metric_registry": joint_registry,
        "thresholds": thresholds,
        "metric_contracts": {
            "boolean_incidence": {
                "value_domain": [0, 1],
                "formula": "weighted_nonzero_incidence_ratio",
                "threshold": "incidence_ratio",
            },
            "rare_incidence": {
                "value_domain": "finite_numeric",
                "formula": "weighted_nonzero_incidence_ratio",
                "threshold": "incidence_ratio",
            },
            "monetary_sign_separated": {
                "value_domain": "finite_numeric",
                "legs": ["positive", "negative"],
                "formula": "incidence_ratio_then_conditional_absolute_quantiles",
                "thresholds": [
                    "incidence_ratio",
                    "conditional_quantile_envelope",
                ],
            },
            "categorical_tvd": {
                "value_domain": "hashable_finite_scalar",
                "formula": "weighted_category_share_total_variation",
                "threshold": "categorical_total_variation",
            },
        },
        "authority_binding": {
            "authority_id": str(authority_receipt["authority_id"]),
            "version": int(authority_receipt["version"]),
        },
    }
