"""Narrow bundle-authority adapter gates for the US pool runtime."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from microcosm.build.spec_engine import (
    RuntimeAuthorities,
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.spec_engine.compiler_ir import CompiledSpecIR
from microcosm.build.spec_engine.model import FrozenMap
from microcosm.build.us_runtime.spec_authority import (
    NodePort,
    NodeQuery,
    USAuthorityProjection,
    USSpecAuthority,
    USSpecAuthorityError,
    compile_us_spec_authority,
)


@pytest.fixture(scope="module")
def compiled_us() -> CompiledSpecIR:
    return compile_spec(load_bundle("us"))


@pytest.fixture(scope="module")
def runtime_authorities(compiled_us: CompiledSpecIR) -> RuntimeAuthorities:
    return compile_runtime_authorities(compiled_us)


@pytest.fixture(scope="module")
def authority(runtime_authorities: RuntimeAuthorities) -> USSpecAuthority:
    return compile_us_spec_authority(runtime_authorities)


def test_adapter_exposes_only_frozen_compiler_authorities(
    compiled_us: CompiledSpecIR,
    runtime_authorities: RuntimeAuthorities,
    authority: USSpecAuthority,
) -> None:
    behavior = compiled_us.runtime_authorities["surfaces"]
    assert isinstance(behavior, FrozenMap)
    expected_behavior = behavior["behavior"]
    assert isinstance(expected_behavior, FrozenMap)
    assert authority.behavior_resources == expected_behavior
    assert authority.authority_sha256 == runtime_authorities.authority_sha256
    assert authority.declared_sources is runtime_authorities.declared_sources
    assert authority.declared_source("acs_household")["acquisition"]["filename"] == (
        "csv_hus.zip"
    )
    assert authority.spec_sha256 == runtime_authorities.spec_binding.spec_sha256
    assert authority.generated_authorities is runtime_authorities.generated_authorities
    assert authority.vintage_authorities is runtime_authorities.vintage_authorities
    assert authority.execution_abi is runtime_authorities.execution_abi
    assert authority.seed_stream_map is runtime_authorities.seed_stream_map
    assert not hasattr(authority, "compiled")
    assert not hasattr(authority, "normalized_resources")
    with pytest.raises(TypeError):
        authority.behavior_resource("spine")["assembly"] = "changed"  # type: ignore[index]


def test_adapter_capability_seal_rejects_divergent_surfaces(
    authority: USSpecAuthority,
) -> None:
    with pytest.raises(USSpecAuthorityError, match="capability seal"):
        replace(authority, _declared_sources=FrozenMap())


def test_adapter_exposes_all_pool_authority_projections_directly(
    runtime_authorities: RuntimeAuthorities,
    authority: USSpecAuthority,
) -> None:
    for projection in USAuthorityProjection:
        assert authority.projection(projection) is runtime_authorities.projection(
            projection.value
        )
    assert authority.publication is runtime_authorities.projection("publication")
    assert authority.sampling is runtime_authorities.projection("sampling")
    assert authority.imputation is runtime_authorities.projection("imputation")
    assert authority.take_up is runtime_authorities.projection("take_up")
    assert authority.battery is runtime_authorities.projection("battery")
    assert authority.battery_components is runtime_authorities.projection(
        "battery_components"
    )
    assert authority.stacked_authority is runtime_authorities.projection(
        "stacked_authority"
    )
    assert authority.stacked_checkpoint_static_components is (
        runtime_authorities.projection("stacked_checkpoint_static_components")
    )


def test_node_queries_use_typed_fields_and_fail_closed(
    authority: USSpecAuthority,
) -> None:
    output_counts = Counter(
        (
            output["entity"],
            output["column"],
            output["coverage_scope"],
        )
        for node in authority.nodes
        for output in node.outputs
    )
    node = next(
        candidate
        for candidate in authority.nodes
        if any(
            output_counts[
                (
                    output["entity"],
                    output["column"],
                    output["coverage_scope"],
                )
            ]
            == 1
            for output in candidate.outputs
        )
    )
    output = next(
        row
        for row in node.outputs
        if output_counts[(row["entity"], row["column"], row["coverage_scope"])] == 1
    )
    query = NodeQuery(
        kernel_ref=node.kernel_ref,
        output_port=NodePort(
            entity=str(output["entity"]),
            column=str(output["column"]),
            scope=str(output["coverage_scope"]),
        ),
    )
    assert authority.require_node(query) is node

    duplicate_kernel = next(
        kernel
        for kernel, count in Counter(
            candidate.kernel_ref for candidate in authority.nodes
        ).items()
        if count > 1
    )
    with pytest.raises(USSpecAuthorityError, match="exactly one"):
        authority.require_node(NodeQuery(kernel_ref=duplicate_kernel))
    with pytest.raises(USSpecAuthorityError, match="matched 0"):
        authority.require_node(NodeQuery(node_id="not-a-compiled-node"))


def test_seed_site_lookup_is_generic_not_program_specific(
    authority: USSpecAuthority,
) -> None:
    node = next(candidate for candidate in authority.nodes if candidate.seed_sites)
    site = node.seed_sites[0]
    assert node in authority.nodes_matching(NodeQuery(seed_site_id=site.id))
    programs = authority.take_up["programs"]
    assert isinstance(programs, tuple)
    program_variables = {
        row["variable"] for row in programs if isinstance(row, FrozenMap)
    }
    assert program_variables
    assert program_variables.isdisjoint(dir(authority))


def test_adapter_rejects_non_capabilities_and_non_authoritative_attestation(
    compiled_us: CompiledSpecIR,
    runtime_authorities: RuntimeAuthorities,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="RuntimeAuthorities"):
        compile_us_spec_authority(compiled_us)  # type: ignore[arg-type]

    fake_provenance = SimpleNamespace(
        to_wire=lambda: {
            "identity_generation": 1,
            "spec_binding": {"attestation": "mirror-attested"},
        }
    )
    monkeypatch.setattr(
        RuntimeAuthorities,
        "run_provenance_identity",
        lambda self, **_kwargs: fake_provenance,
    )
    with pytest.raises(USSpecAuthorityError, match="bundle-authoritative"):
        compile_us_spec_authority(runtime_authorities)


def test_adapter_rejects_invalid_query_shapes(authority: USSpecAuthority) -> None:
    with pytest.raises(USSpecAuthorityError, match="at least one"):
        NodeQuery()
    with pytest.raises(USSpecAuthorityError, match="kernel:"):
        NodeQuery(kernel_ref="untyped-kernel")
    with pytest.raises(USSpecAuthorityError, match="unknown"):
        authority.projection("unsealed-projection")
    with pytest.raises(USSpecAuthorityError, match="no object resource"):
        authority.behavior_resource("schema")
