from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest

from microcosm.build.spec_engine import SpecValidationError, load_schema_registry
from microcosm.build.spec_engine.identity_contracts import (
    IdentityContractError,
    resolve_seed_site_bindings,
)
from microcosm.build.spec_engine.model import SeedSiteOwnerKind
from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL
from microcosm.build.us_runtime.h5_io import US_STACKED_POOL_OPERATOR_ORDER
from microcosm.build.us_runtime.multispine_pool import (
    POOL_DERIVE_OPERATOR_ORDER,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION,
)
from microcosm.build.us_runtime.qbi_inputs import (
    us_qbi_reconciliation_contract_identity,
)
from microcosm.build.us_runtime.qbi_passive_passthrough import (
    us_qbi_passive_passthrough_contract_identity,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_TRANSFER_GROUPS,
)
from tools.build_us_multispine_pool import (
    stacked_checkpoint_artifact_protocol_identity,
)
from tools.us_bundle_generation.core import build_sources, build_spine
from tools.us_bundle_generation.identity_contracts import build_pipeline_contract


@pytest.fixture(scope="module")
def identity_documents() -> tuple[dict[str, object], list[dict[str, object]]]:
    sources = build_sources()
    return build_spine(), sources["stages"]


def _resolve(spine: Mapping[str, object], source_stages: list[dict[str, object]]):
    return resolve_seed_site_bindings(
        spine,
        protocol=LEGACY_V1_PROTOCOL,
        source_stage_ids=frozenset(str(row["stage"]) for row in source_stages),
        producer_node_ids=frozenset(
            {
                "primary_puf_qrf",
                *(group.name for group in CANONICAL_US_LATE_TRANSFER_GROUPS),
            }
        ),
    )


def test_pipeline_contract_is_an_exact_generation_zero_projection() -> None:
    contract = build_pipeline_contract()
    assert (
        contract["artifact_protocol"]
        == (stacked_checkpoint_artifact_protocol_identity())
        == {
            "artifact_kind": "populace_us_stacked_pool_checkpoint_identity",
            "schema_version": 1,
            "materializer_version": 11,
            "pipeline": "us-stacked-pool",
        }
    )
    assert contract["stacked_operator_order"] == list(US_STACKED_POOL_OPERATOR_ORDER)
    assert contract["pre_clone_source_operator_order"] == list(
        POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert contract["post_clone_source_operator_order"] == list(
        POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert contract["derive_operator_order"] == list(POOL_DERIVE_OPERATOR_ORDER)
    assert contract["qbi_passive_passthrough"] == (
        us_qbi_passive_passthrough_contract_identity()
    )
    assert contract["qbi_reconciliation"] == (us_qbi_reconciliation_contract_identity())
    assert contract["simulation_household_batch_size"] == {
        "value": POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
        "classification": "normative_legacy_identity_fence",
        "current_identity_effect": "bound_into_generation_0_checkpoint_identity",
    }


def test_tail_artifact_manifest_schema_has_one_typed_bundle_home(
    identity_documents: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    spine, _ = identity_documents
    tail_support = spine["support_roles"][0]["tail_support"]
    assert (
        tail_support["manifest_schema_version"]
        == (PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION)
        == 2
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda spine: spine["pipeline_contract"].update({"undeclared": True}),
        lambda spine: spine["seed_site_bindings"][0]["owners"][0].update(
            {"undeclared": True}
        ),
        lambda spine: spine["support_roles"][0]["tail_support"].update(
            {"undeclared": True}
        ),
    ],
)
def test_identity_contract_objects_are_closed_world(
    identity_documents: tuple[dict[str, object], list[dict[str, object]]], mutate
) -> None:
    spine, _ = identity_documents
    mutated = copy.deepcopy(spine)
    mutate(mutated)

    with pytest.raises(SpecValidationError, match="undeclared"):
        load_schema_registry().validate(mutated, "spine.schema.json")


def test_all_53_seed_sites_resolve_to_typed_real_owners(
    identity_documents: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    spine, source_stages = identity_documents
    bindings = _resolve(spine, source_stages)

    assert len(bindings) == len(LEGACY_V1_PROTOCOL.sites) == 53
    assert [binding.site for binding in bindings] == [
        site.id for site in LEGACY_V1_PROTOCOL.sites
    ]
    by_site = {binding.site: binding for binding in bindings}
    assert len(by_site["acs_qrf_fit_draw"].owners) == 20
    assert {owner.kind for owner in by_site["acs_qrf_fit_draw"].owners} == {
        SeedSiteOwnerKind.PRODUCER_NODE,
        SeedSiteOwnerKind.PIPELINE_OPERATION,
    }
    assert by_site["primary_qrf_fit_draw"].owners[0].id == "primary_puf_qrf"
    assert by_site["puf_clone_attachment"].owners[0].id == (
        "prepare_multispine_source_inputs_for_clone"
    )
    assert by_site["capital_gains_tail_random_rank"].owners[0].id == (
        "prepare_stacked_tail_derivation"
    )
    assert by_site["torch_calibration_reseed"].owners[0].id == "calibrate"


def test_missing_duplicate_unknown_site_or_dangling_owner_is_refused(
    identity_documents: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    spine, source_stages = identity_documents

    missing = copy.deepcopy(spine)
    missing["seed_site_bindings"].pop()
    with pytest.raises(IdentityContractError, match="sites are unbound"):
        _resolve(missing, source_stages)

    duplicate = copy.deepcopy(spine)
    duplicate["seed_site_bindings"][1]["site"] = duplicate["seed_site_bindings"][0][
        "site"
    ]
    with pytest.raises(IdentityContractError, match="duplicate seed site"):
        _resolve(duplicate, source_stages)

    unknown = copy.deepcopy(spine)
    unknown["seed_site_bindings"][0]["site"] = "unknown_site"
    with pytest.raises(IdentityContractError, match="unknown legacy-v1 seed site"):
        _resolve(unknown, source_stages)

    dangling = copy.deepcopy(spine)
    dangling["seed_site_bindings"][0]["owners"][0]["id"] = "missing_operation"
    with pytest.raises(IdentityContractError, match="dangling pipeline_operation"):
        _resolve(dangling, source_stages)


def test_pipeline_contract_and_seed_bindings_are_all_or_nothing(
    identity_documents: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    spine, source_stages = identity_documents
    for field in ("pipeline_contract", "seed_site_bindings"):
        mutated = copy.deepcopy(spine)
        del mutated[field]
        with pytest.raises(IdentityContractError, match="declared together"):
            _resolve(mutated, source_stages)
