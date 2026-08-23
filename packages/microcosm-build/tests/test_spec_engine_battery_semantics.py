"""Golden gates for normalized battery compatibility projections."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass

import pytest

import microcosm.build.us_runtime.stacked_spine as stacked_spine
from microcosm.build.spec_engine import ResourceKind, load_bundle
from microcosm.build.spec_engine.battery_semantics import (
    derive_battery_registry_views,
    project_battery_authority_components,
    project_battery_legacy_contract,
)
from microcosm.build.spec_engine.errors import SpecValidationError
from microcosm.build.us_runtime.stacked_battery_contract import (
    build_live_stacked_battery_contract,
)
from microcosm.build.us_runtime.stacked_spine import (
    CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY,
    CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY,
    CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE,
    CANONICAL_STACKED_DECLARED_SURFACE,
    stacked_spine_authority_receipt,
)
from tools.us_bundle_generation.contracts import build_battery_contract


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_ready(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


@pytest.fixture(scope="module")
def battery() -> dict[str, object]:
    value = load_bundle("us").domain(ResourceKind.BATTERY).to_wire()
    assert isinstance(value, dict)
    return value


def test_battery_registry_views_match_live_registry(
    battery: dict[str, object],
) -> None:
    views = derive_battery_registry_views(battery)
    assert views["declared_surface"] == _json_ready(CANONICAL_STACKED_DECLARED_SURFACE)
    assert views["required_scalar_targets"] == 131
    assert views["required_joint_targets"] == 1
    assert views["metric_counts"] == {
        "boolean_incidence": 48,
        "categorical_tvd": 4,
        "monetary_sign_separated": 79,
    }


def test_battery_authority_components_match_live_constants(
    battery: dict[str, object],
) -> None:
    components = project_battery_authority_components(battery)
    expected_metrics = [
        {
            "entity": entity,
            "family": family,
            "column": column,
            "clone_index": clone_index,
            "metric": CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY[
                (entity, family, column, clone_index)
            ],
        }
        for entity, family, column, clone_index in sorted(
            CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
        )
    ]
    expected_joint = [
        {
            "entity": entity,
            "family": family,
            "columns": list(columns),
            "clone_index": clone_index,
            "metric": CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY[
                (entity, family, columns, clone_index)
            ],
        }
        for entity, family, columns, clone_index in sorted(
            CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY
        )
    ]
    profile = CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE
    assert components == {
        "declared_surface": _json_ready(CANONICAL_STACKED_DECLARED_SURFACE),
        "metric_registry": expected_metrics,
        "joint_metric_registry": expected_joint,
        "support_profile": {
            "min_effective_support": profile.min_effective_support,
            "profile_id": profile.profile_id,
            "version": profile.version,
        },
    }


def test_battery_legacy_projection_survives_resolved_normalization(
    battery: dict[str, object],
) -> None:
    authority = stacked_spine_authority_receipt()
    projected = project_battery_legacy_contract(
        battery,
        authority_receipt=authority,
    )
    live_contract = build_live_stacked_battery_contract()
    assert build_battery_contract() == live_contract
    raw_projection = project_battery_legacy_contract(
        live_contract,
        authority_receipt=authority,
    )
    assert projected == raw_projection
    assert projected["authority_binding"]["expected_sha256"] == authority["sha256"]
    assert projected["gates"][0]["thresholds"] == {
        "absolute": 0.0,
        "relative": 0.0,
    }
    assert projected["gates"][1]["metric"]["params"] == {
        **battery["gates"][1]["metric"]["params"],
        "required_joint_targets": 1,
        "required_scalar_targets": 131,
    }


def test_battery_registry_projection_refuses_duplicate_keys(
    battery: dict[str, object],
) -> None:
    mutated = deepcopy(battery)
    mutated["metric_registry"].append(deepcopy(mutated["metric_registry"][0]))
    with pytest.raises(SpecValidationError, match="duplicate metric key"):
        derive_battery_registry_views(mutated)


def test_battery_registry_projection_accepts_nonzero_clone_role(
    battery: dict[str, object],
) -> None:
    mutated = deepcopy(battery)
    row = next(
        item for item in mutated["metric_registry"] if item["clone_index"] == 0
    )
    row["clone_index"] = 1

    views = derive_battery_registry_views(mutated)

    assert views["declared_surface"] == _json_ready(CANONICAL_STACKED_DECLARED_SURFACE)


def test_battery_registry_projection_refuses_physical_target_across_roles(
    battery: dict[str, object],
) -> None:
    mutated = deepcopy(battery)
    duplicate = deepcopy(mutated["metric_registry"][0])
    duplicate["clone_index"] = int(duplicate["clone_index"]) + 1
    mutated["metric_registry"].append(duplicate)

    with pytest.raises(SpecValidationError, match="duplicate physical target"):
        derive_battery_registry_views(mutated)


def test_live_battery_contract_emits_registered_nonzero_clone_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = stacked_spine_authority_receipt()
    mutated = dict(CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    source_key = next(key for key in sorted(mutated) if key[3] == 0)
    entity, family, column, _clone_index = source_key
    metric = mutated.pop(source_key)
    mutated[(entity, family, column, 1)] = metric
    monkeypatch.setattr(
        stacked_spine,
        "CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY",
        mutated,
    )
    monkeypatch.setattr(
        stacked_spine,
        "stacked_spine_authority_receipt",
        lambda: authority,
    )

    contract = build_live_stacked_battery_contract()
    rows = [
        row
        for row in contract["metric_registry"]
        if (row["entity"], row["family"], row["column"])
        == (entity, family, column)
    ]

    assert len(rows) == 1
    assert rows[0]["clone_index"] == 1


def test_live_battery_contract_refuses_physical_target_across_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated = dict(CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    source_key = next(iter(mutated))
    entity, family, column, clone_index = source_key
    duplicate_role = clone_index + 1
    mutated[(entity, family, column, duplicate_role)] = mutated[source_key]
    monkeypatch.setattr(
        stacked_spine,
        "CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY",
        mutated,
    )

    with pytest.raises(RuntimeError, match="duplicate physical target"):
        build_live_stacked_battery_contract()
