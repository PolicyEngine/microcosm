"""Pure compatibility projections for normalized gate-battery contracts.

The battery YAML owns metric membership and ordering.  Constants-era readers
also expect redundant counts, a nested declared surface, and authority digest
bindings.  This module derives those views without importing the US runtime or
the one-shot bundle generator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy

from .errors import SpecValidationError


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{location}: object required")
    return value


def _array(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise SpecValidationError(f"{location}: array required")
    return value


def derive_battery_registry_views(document: Mapping[str, object]) -> dict[str, object]:
    """Derive constants-era counts and declared surface from metric registries."""

    registry = [
        _mapping(value, "battery/metric_registry row")
        for value in _array(document.get("metric_registry"), "battery/metric_registry")
    ]
    joint_registry = [
        _mapping(value, "battery/joint_metric_registry row")
        for value in _array(
            document.get("joint_metric_registry"),
            "battery/joint_metric_registry",
        )
    ]
    surface: dict[str, dict[str, list[str]]] = {}
    seen: set[tuple[str, str, str, int]] = set()
    for index, row in enumerate(registry):
        try:
            key = (
                str(row["entity"]),
                str(row["family"]),
                str(row["column"]),
                int(row["clone_index"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SpecValidationError(
                f"battery/metric_registry/{index}: malformed registry row"
            ) from error
        if key in seen:
            raise SpecValidationError(
                f"battery/metric_registry/{index}: duplicate metric key {key!r}"
            )
        seen.add(key)
        if key[3] != 0:
            raise SpecValidationError(
                "battery/metric_registry/"
                f"{index}/clone_index: generation-0 declared surface requires 0"
            )
        surface.setdefault(key[0], {}).setdefault(key[1], []).append(key[2])

    metric_counts = dict(
        sorted(Counter(str(row["metric"]) for row in registry).items())
    )
    return {
        "completeness": {
            "targets": len(registry),
            "source": "declared_surface",
        },
        "metric_counts": metric_counts,
        "declared_surface": surface,
        "required_scalar_targets": len(registry),
        "required_joint_targets": len(joint_registry),
    }


def project_battery_authority_components(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Return the battery-owned stacked-authority component payloads.

    The complete eight-component stacked receipt is assembled elsewhere.  The
    four values returned here are entirely owned by the battery domain and
    retain the constants-era component ordering.
    """

    derived = derive_battery_registry_views(document)
    metric_registry = [
        deepcopy(dict(_mapping(value, "battery/metric_registry row")))
        for value in _array(document.get("metric_registry"), "battery/metric_registry")
    ]
    metric_registry.sort(
        key=lambda row: (
            str(row["entity"]),
            str(row["family"]),
            str(row["column"]),
            int(row["clone_index"]),
        )
    )
    joint_registry = [
        deepcopy(dict(_mapping(value, "battery/joint_metric_registry row")))
        for value in _array(
            document.get("joint_metric_registry"),
            "battery/joint_metric_registry",
        )
    ]
    joint_registry.sort(
        key=lambda row: (
            str(row["entity"]),
            str(row["family"]),
            tuple(str(value) for value in _array(row["columns"], "joint columns")),
            int(row["clone_index"]),
        )
    )
    profile = _mapping(document.get("support_profile"), "battery/support_profile")
    return {
        "declared_surface": deepcopy(derived["declared_surface"]),
        "metric_registry": metric_registry,
        "joint_metric_registry": joint_registry,
        "support_profile": {
            "min_effective_support": profile["min_effective_support"],
            "profile_id": profile["profile_id"],
            "version": profile["version"],
        },
    }


def project_battery_legacy_contract(
    document: Mapping[str, object],
    *,
    authority_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Inflate compiler-derived fields expected by constants-era consumers."""

    result = deepcopy(dict(document))
    derived = derive_battery_registry_views(document)
    result["completeness"] = derived["completeness"]
    result["metric_counts"] = derived["metric_counts"]
    result["declared_surface"] = derived["declared_surface"]
    scalar_count = int(derived["required_scalar_targets"])
    joint_count = int(derived["required_joint_targets"])

    for index, gate_value in enumerate(_array(result.get("gates"), "battery/gates")):
        gate = _mapping(gate_value, f"battery/gates/{index}")
        metric = _mapping(gate.get("metric"), f"battery/gates/{index}/metric")
        params = _mapping(metric.get("params"), f"battery/gates/{index}/metric/params")
        params["required_scalar_targets"] = scalar_count  # type: ignore[index]
        params["required_joint_targets"] = (  # type: ignore[index]
            joint_count if gate.get("id") == "us_by_origin_battery" else 0
        )
        if gate.get("id") == "us_stacked_completeness":
            thresholds = _mapping(
                gate.get("thresholds"), f"battery/gates/{index}/thresholds"
            )
            for key in ("absolute", "relative"):
                value = thresholds.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SpecValidationError(
                        f"battery/gates/{index}/thresholds/{key}: number required"
                    )
                thresholds[key] = float(value)  # type: ignore[index]
        reference = _mapping(gate.get("reference"), f"battery/gates/{index}/reference")
        if dict(reference) != {
            "kind": "compiled_authority",
            "authority_ref": "authority_binding",
        }:
            raise SpecValidationError(
                f"battery/gates/{index}/reference: unsupported authority reference"
            )
        gate["reference"] = {  # type: ignore[index]
            "kind": "data_digest",
            "sha256": authority_receipt["sha256"],
        }

    binding = _mapping(result.get("authority_binding"), "battery/authority_binding")
    if binding.get("authority_id") != authority_receipt.get(
        "authority_id"
    ) or binding.get("version") != authority_receipt.get("version"):
        raise SpecValidationError(
            "battery/authority_binding: identity differs from compiled receipt"
        )
    binding["expected_sha256"] = authority_receipt["sha256"]  # type: ignore[index]
    components = _mapping(
        authority_receipt.get("components"), "stacked authority/components"
    )
    binding["components"] = [  # type: ignore[index]
        {
            "id": component_id,
            "expected_sha256": _mapping(
                component, f"stacked authority/components/{component_id}"
            )["sha256"],
        }
        for component_id, component in sorted(components.items())
    ]
    return result


__all__ = [
    "derive_battery_registry_views",
    "project_battery_authority_components",
    "project_battery_legacy_contract",
]
