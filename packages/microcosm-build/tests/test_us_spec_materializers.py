"""Compiler-authoritative source materializers for the US runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from microcosm.build.spec_engine import (
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.spec_engine.brokers import DeclaredSource
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.model import FrozenMap, freeze_json, thaw_json
from microcosm.build.us_runtime import acs_sources
from microcosm.build.us_runtime.pool_runtime_plan import (
    SourceAuthority,
    USPoolRuntimePlan,
)
from microcosm.build.us_runtime.spec_authority import (
    USSpecAuthority,
    USSpecAuthorityError,
    _capability_sha256,
    compile_us_spec_authority,
)
from microcosm.build.us_runtime.spec_materializers import (
    SpecMaterializationError,
    compile_declared_source_pins,
    materialize_acs_source_manifest,
)


@pytest.fixture(scope="module")
def authority() -> USSpecAuthority:
    return compile_us_spec_authority(
        compile_runtime_authorities(compile_spec(load_bundle("us")))
    )


def _sources(authority: USSpecAuthority) -> SourceAuthority:
    return USPoolRuntimePlan.from_spec_authority(authority).sources


def _reseal_authority(
    authority: USSpecAuthority,
    **changes: object,
) -> USSpecAuthority:
    surfaces = {
        "_behavior": authority._behavior,
        "_projections": authority._projections,
        "_declared_sources": authority._declared_sources,
        "_generated_authorities": authority._generated_authorities,
        "_vintage_authorities": authority._vintage_authorities,
        "_execution_abi": authority._execution_abi,
        "_seed_stream_map": authority._seed_stream_map,
        "_nodes": authority._nodes,
    }
    surfaces.update(changes)
    seal = _capability_sha256(
        authority_sha256=authority.authority_sha256,
        spec_sha256=authority.spec_sha256,
        identity_generation=authority.identity_generation,
        behavior=surfaces["_behavior"],
        projections=surfaces["_projections"],
        declared_sources=surfaces["_declared_sources"],
        generated_authorities=surfaces["_generated_authorities"],
        vintage_authorities=surfaces["_vintage_authorities"],
        execution_abi=surfaces["_execution_abi"],
        seed_stream_map=surfaces["_seed_stream_map"],
        nodes=surfaces["_nodes"],
    )
    return replace(authority, **changes, _seal_sha256=seal)


def _mutate_declared_sources(
    authority: USSpecAuthority,
    mutation: Callable[[dict[str, Any]], None],
    *,
    reseal: bool = True,
) -> USSpecAuthority:
    wire = thaw_json(authority.declared_sources)
    assert isinstance(wire, dict)
    mutation(cast(dict[str, Any], wire))
    if reseal:
        wire["sha256"] = sha256_json(
            {
                "schema_version": wire["schema_version"],
                "sources": wire["sources"],
            }
        )
    frozen = freeze_json(wire)
    assert isinstance(frozen, FrozenMap)
    return _reseal_authority(authority, _declared_sources=frozen)


def _mutate_behavior(
    authority: USSpecAuthority,
    mutation: Callable[[dict[str, Any]], None],
) -> USSpecAuthority:
    wire = thaw_json(authority.behavior_resources)
    assert isinstance(wire, dict)
    mutation(cast(dict[str, Any], wire))
    frozen = freeze_json(wire)
    assert isinstance(frozen, FrozenMap)
    return _reseal_authority(authority, _behavior=frozen)


def test_narrow_capability_refuses_unsealed_surface_replacement(
    authority: USSpecAuthority,
) -> None:
    with pytest.raises(USSpecAuthorityError, match="capability seal"):
        replace(authority, _declared_sources=FrozenMap())


def test_acs_manifest_is_exact_legacy_value_without_calling_loader(
    authority: USSpecAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constants_oracle = acs_sources.load_acs_source_manifest()

    def forbidden_loader(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("bundle materialization called the constants loader")

    monkeypatch.setattr(acs_sources, "load_acs_source_manifest", forbidden_loader)
    materialized = materialize_acs_source_manifest(_sources(authority))

    assert materialized == constants_oracle
    assert materialized.artifact("household") == constants_oracle.artifact("household")
    assert materialized.artifact("person") == constants_oracle.artifact("person")


def test_declared_source_pins_are_path_free_and_file_broker_ready(
    authority: USSpecAuthority,
    tmp_path: Path,
) -> None:
    pins = compile_declared_source_pins(_sources(authority))
    household = pins.require("acs_household")

    assert pins.authority_sha256 == authority.declared_sources["sha256"]
    assert household.broker_identity_wire() == {
        "source_id": "acs_household",
        "content_sha256": (
            "8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0"
        ),
        "byte_size": 251500587,
    }
    assert "path" not in household.broker_identity_wire()
    assert "url" not in household.broker_identity_wire()

    source_path = tmp_path / "household.zip"
    source_path.write_bytes(b"broker verifies these bytes on first read")
    bound = pins.bind(
        household.id,
        source_path,
        supplied_sha256=household.sha256,
        supplied_byte_size=household.byte_size,
    )
    assert isinstance(bound, DeclaredSource)
    assert bound.id == household.id
    assert bound.sha256 == household.sha256
    assert bound.byte_size == household.byte_size
    assert bound.path == source_path

    with pytest.raises(SpecMaterializationError, match="differs"):
        pins.bind(
            household.id,
            source_path,
            supplied_sha256="0" * 64,
        )
    with pytest.raises(SpecMaterializationError, match="no byte_size"):
        pins.require("us_puma_ladder_2020").broker_identity_wire()


def test_declared_source_digest_covers_documentation_acquisition_fields(
    authority: USSpecAuthority,
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], value["sources"])
        cast(dict[str, Any], rows[1]["acquisition"])["verified_on"] = "2026-08-19"

    unsealed = _mutate_declared_sources(authority, mutate, reseal=False)
    with pytest.raises(SpecMaterializationError, match="digest"):
        compile_declared_source_pins(_sources(unsealed))

    resealed = _mutate_declared_sources(authority, mutate)
    assert compile_declared_source_pins(_sources(resealed)).authority_sha256 != (
        compile_declared_source_pins(_sources(authority)).authority_sha256
    )


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_acs_manifest_rejects_missing_or_extra_acquisition_records(
    authority: USSpecAuthority,
    mode: str,
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], value["sources"])
        if mode == "missing":
            rows[2].pop("acquisition")
        else:
            rows[0]["acquisition"] = dict(rows[1]["acquisition"])

    changed = _mutate_declared_sources(authority, mutate)
    with pytest.raises(SpecMaterializationError, match="exactly"):
        materialize_acs_source_manifest(_sources(changed))


@pytest.mark.parametrize("mode", ["directory", "swapped"])
def test_acs_manifest_rejects_inconsistent_acquisition_records(
    authority: USSpecAuthority,
    mode: str,
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], value["sources"])
        if mode == "swapped":
            rows[1]["acquisition"], rows[2]["acquisition"] = (
                rows[2]["acquisition"],
                rows[1]["acquisition"],
            )
            return
        acquisition = cast(dict[str, Any], rows[2]["acquisition"])
        acquisition["source_directory"] = "https://example.invalid/2024/1-Year/"
        acquisition["url"] = f"{acquisition['source_directory']}csv_pus.zip"

    changed = _mutate_declared_sources(authority, mutate)
    pattern = "filename must" if mode == "swapped" else "share source_directory"
    with pytest.raises(SpecMaterializationError, match=pattern):
        materialize_acs_source_manifest(_sources(changed))


def test_acs_vintage_refuses_drift_from_resolved_compiler_authority(
    authority: USSpecAuthority,
) -> None:
    def change_vintage_authority(value: dict[str, Any]) -> None:
        sources = cast(dict[str, Any], value["sources"])
        rows = cast(list[dict[str, Any]], sources["sources"])
        vintage_rows = cast(list[dict[str, Any]], rows[1]["vintage_authorities"])
        vintage_rows[0]["value"] = 2025

    def change_acquisition_vintage(value: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], value["sources"])
        for row in rows[1:3]:
            acquisition = cast(dict[str, Any], row["acquisition"])
            acquisition["source_directory"] = cast(
                str, acquisition["source_directory"]
            ).replace("/2024/", "/2025/")
            acquisition["url"] = cast(str, acquisition["url"]).replace(
                "/2024/", "/2025/"
            )

    changed = _mutate_behavior(authority, change_vintage_authority)
    changed = _mutate_declared_sources(changed, change_acquisition_vintage)
    with pytest.raises(
        SpecMaterializationError,
        match="source and resolved authority values differ",
    ):
        materialize_acs_source_manifest(_sources(changed))


def test_acs_vintage_refuses_inconsistent_frozen_source_authority(
    authority: USSpecAuthority,
) -> None:
    def mutate(value: dict[str, Any]) -> None:
        sources = cast(dict[str, Any], value["sources"])
        rows = cast(list[dict[str, Any]], sources["sources"])
        rows[1]["sha256"] = "0" * 64

    changed = _mutate_behavior(authority, mutate)
    with pytest.raises(SpecMaterializationError, match="content pin is inconsistent"):
        materialize_acs_source_manifest(_sources(changed))


def test_acs_vintage_refuses_resolved_authority_digest_drift(
    authority: USSpecAuthority,
) -> None:
    wire = thaw_json(authority.vintage_authorities)
    assert isinstance(wire, dict)
    records = cast(dict[str, dict[str, Any]], wire["records"])
    records["acs_2024"]["authority_sha256"] = "0" * 64
    changed_vintages = freeze_json(wire)
    assert isinstance(changed_vintages, FrozenMap)
    changed = _reseal_authority(
        authority,
        _vintage_authorities=changed_vintages,
    )

    with pytest.raises(SpecMaterializationError, match="digest differs"):
        materialize_acs_source_manifest(_sources(changed))


def test_source_materializers_require_source_authority() -> None:
    with pytest.raises(TypeError, match="SourceAuthority"):
        compile_declared_source_pins(object())  # type: ignore[arg-type]
