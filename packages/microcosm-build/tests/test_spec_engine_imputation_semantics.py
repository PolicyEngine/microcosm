"""Golden gates for the production imputation compatibility projector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import pytest

from microcosm.build.spec_engine import ResourceKind, load_bundle
from microcosm.build.spec_engine.canonical import canonical_json_bytes
from microcosm.build.spec_engine.imputation_semantics import (
    derive_primary_effective_predictor_tuples,
    project_imputation_legacy_payloads,
)
from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL
from microcosm.build.us_runtime.acs_transfer import (
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    acs_transfer_execution_contract_identity,
)
from microcosm.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
    PRIMARY_QRF_TARGET_ORDER,
    PRIMARY_QRF_TARGET_ORDER_SHA256,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.build.us_runtime.stacked_spine import (
    stacked_gap_fill_plan,
    stacked_gap_fill_producer_schedule_receipt,
    stacked_late_producer_resource_semantics_receipt,
)
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    us_late_overlap_ownership_receipt,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    us_late_producer_schedule_receipt,
)


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_ready(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_ready(child) for child in value)
    return value


@pytest.fixture(scope="module")
def us_domains() -> dict[str, dict[str, object]]:
    spec = load_bundle("us")
    result = {
        kind.value: spec.domain(kind).to_wire()
        for kind in (
            ResourceKind.BUNDLE,
            ResourceKind.IMPUTATION,
            ResourceKind.SOURCES,
            ResourceKind.SPINE,
        )
    }
    assert all(isinstance(value, dict) for value in result.values())
    return result  # type: ignore[return-value]


@pytest.fixture(scope="module")
def projected(us_domains: dict[str, dict[str, object]]) -> dict[str, object]:
    return project_imputation_legacy_payloads(
        us_domains["imputation"],
        sources_document=us_domains["sources"],
        spine_document=us_domains["spine"],
        bundle_document=us_domains["bundle"],
    )


def test_imputation_projector_matches_live_plans_and_graph_receipts(
    projected: dict[str, object],
) -> None:
    live = {
        "gap_fill_plan": _json_ready(stacked_gap_fill_plan()),
        "gap_fill_producer_schedule_receipt": _json_ready(
            stacked_gap_fill_producer_schedule_receipt()
        ),
        "late_producer_schedule_receipt": _json_ready(
            us_late_producer_schedule_receipt()
        ),
        "overlap_ownership": _json_ready(us_late_overlap_ownership_receipt()),
    }
    for key, expected in live.items():
        assert canonical_json_bytes(projected[key]) == canonical_json_bytes(expected)
    assert projected["late_producer_schedule_receipt"]["schedule_sha256"] == (
        "dcf3c6d2eade3449836c49a1dc4d3b8cd395aab9142db700c3c60598fa9c1c79"
    )
    assert projected["overlap_ownership"]["sha256"] == (
        "5f64f0aac49e2313177564f71876bffc8c81b3ded4df701e70930e60e9c98356"
    )


def test_imputation_projector_matches_live_primary_chain(
    projected: dict[str, object],
    us_domains: dict[str, dict[str, object]],
) -> None:
    assert projected["primary_qrf"] == {
        "predictors": list(PUF_TAX_DETAIL_DEFAULT_PREDICTORS),
        "person_outputs": list(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS),
        "tax_unit_outputs": list(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS),
        "target_order": list(PRIMARY_QRF_TARGET_ORDER),
        "target_order_sha256": PRIMARY_QRF_TARGET_ORDER_SHA256,
        "target_order_digest_rule": "sha256(compact_json_array)",
        "checkpoint_schema_version": PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
    }
    tuples = derive_primary_effective_predictor_tuples(us_domains["imputation"])
    assert len(tuples) == 65
    preceding: list[str] = []
    for row, target in zip(tuples, PRIMARY_QRF_TARGET_ORDER, strict=True):
        assert row["target"] == target
        assert row["predictors"] == [*PUF_TAX_DETAIL_DEFAULT_PREDICTORS, *preceding]
        preceding.append(target)


def test_imputation_projector_matches_live_resource_semantics(
    projected: dict[str, object],
    us_domains: dict[str, dict[str, object]],
) -> None:
    imputation = us_domains["imputation"]
    spine = us_domains["spine"]
    role = next(row for row in spine["support_roles"] if row["id"] == "puf_tax_detail")
    attachment = role["attachment"]
    model_seed = LEGACY_V1_PROTOCOL.site("primary_qrf_fit_draw").default
    params = imputation["models"]["regime_gated_qrf"]["params"]
    live = _json_ready(
        stacked_late_producer_resource_semantics_receipt(
            clone_attachment_fraction=attachment["fraction"]["default"],
            clone_attachment_seed=attachment["seed"]["default"],
            primary_seed=model_seed,
            primary_n_estimators=params["n_estimators"],
            transfer_seed=model_seed,
            transfer_n_estimators=params["n_estimators"],
            transfer_max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
        )
    )
    assert projected["late_producer_resource_semantics"] == live


def test_imputation_projector_matches_live_transfer_contracts(
    projected: dict[str, object],
) -> None:
    contracts = projected["transfer_execution_contract_identities"]
    assert contracts["base"] == acs_transfer_execution_contract_identity(
        targets=[], derive_schedule_d=False
    )
    directions = list(stacked_gap_fill_plan())
    assert len(contracts["early"]) == len(directions) == 2
    for row, direction in zip(contracts["early"], directions, strict=True):
        targets = [
            target
            for families in direction.target_families.values()
            for family_targets in families.values()
            for target in family_targets
        ]
        assert row["identity"] == acs_transfer_execution_contract_identity(
            targets=targets,
            derive_schedule_d=True,
        )
    live_resources = projected["late_producer_resource_semantics"]
    by_producer = {row["producer"]: row for row in live_resources["producers"]}
    assert len(contracts["late"]) == len(CANONICAL_US_LATE_TRANSFER_GROUPS) == 19
    for row, group in zip(
        contracts["late"], CANONICAL_US_LATE_TRANSFER_GROUPS, strict=True
    ):
        binding = by_producer[group.name]["resources"][
            f"{group.entity}.@late_transfer_model_config"
        ]["binding"]
        assert row["ordered_targets"] == list(group.targets)
        assert row["identity"] == binding["transfer_execution_contract"]
