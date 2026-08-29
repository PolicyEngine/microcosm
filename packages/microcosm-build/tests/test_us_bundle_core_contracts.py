from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from microcosm.build.spec_engine import (
    F0_KERNEL_REGISTRY,
    SpecValidationError,
    load_schema_registry,
)
from microcosm.build.spec_engine.canonical import (
    documentation_envelope,
    normalize_and_project,
    sha256_json,
    spec_envelope,
)
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.spec_engine.resolver import (
    SpecResolutionError,
    resolve_cross_references,
)

Mutation = Callable[[dict[str, Any]], None]
ROOT = Path(__file__).resolve().parents[3]
CORE_PATH = ROOT / "tools/us_bundle_generation/core.py"
CORE_SPEC = importlib.util.spec_from_file_location("f0_us_bundle_core", CORE_PATH)
assert CORE_SPEC is not None and CORE_SPEC.loader is not None
CORE_MODULE = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(CORE_MODULE)
build_catalogs = CORE_MODULE.build_catalogs
build_bundle = CORE_MODULE.build_bundle
build_geography = CORE_MODULE.build_geography
build_publication = CORE_MODULE.build_publication
build_sources = CORE_MODULE.build_sources
build_vintages = CORE_MODULE.build_vintages
ENGINE_LOCK = json.loads(
    (
        ROOT / "packages/microcosm-build/src/microcosm/build/us/engine_abi.lock.json"
    ).read_text(encoding="utf-8")
)


def _vintage_resources() -> dict[str, Any]:
    return {
        "bundle": build_bundle(),
        "sources": build_sources(),
        "vintages": build_vintages(),
        "publication": build_publication(),
        "imputation": {},
    }


def _resolve_vintages(
    resources: dict[str, Any], *, engine_lock: dict[str, Any] = ENGINE_LOCK
):
    return resolve_cross_references(
        resources,
        kernel_registry=F0_KERNEL_REGISTRY,
        generated_authorities={"engine_abi_lock": engine_lock},
    )


def _source_surface_hashes(
    sources: dict[str, Any],
) -> tuple[str, str, dict[str, object]]:
    registry = load_schema_registry()
    normalized = registry.validate_and_inject_defaults(sources, "sources.schema.json")
    _, frozen_projections = normalize_and_project(
        normalized,
        schema_id="sources.schema.json",
        registry=registry,
    )
    projections = thaw_json(frozen_projections)
    normative = {"spec/sources.yaml": projections["normative"]}
    documentation = {"spec/sources.yaml": projections["documentation"]}
    return (
        sha256_json(
            spec_envelope(country="us", schema_version=1, normative_files=normative)
        ),
        sha256_json(
            documentation_envelope(
                country="us",
                schema_version=1,
                documentation_files=documentation,
            )
        ),
        projections,
    )


def _walk_values(value: object, path: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            rows.append((child_path, child))
            rows.extend(_walk_values(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_walk_values(child, f"{path}/{index}"))
    return rows


def test_source_surface_classification_is_complete() -> None:
    sources = build_sources()
    _, _, surfaces = _source_surface_hashes(sources)
    normative = surfaces["normative"]
    operational = surfaces["operational"]
    first_pin = normative["sources"][0]
    assert first_pin == {
        "byte_size": sources["sources"][0]["byte_size"],
        "id": "asec_raw_stage",
        "loader": "kernel:load_asec_raw_stage_checkpoint",
        "role": "asec_raw_stage",
        "sha256": sources["sources"][0]["sha256"],
        "vintages": ["vintage:asec_2024", "vintage:asec_2023"],
        "vintage_authorities": [
            {"id": "asec_2022", "kind": "survey_period", "value": 2022},
            {"id": "asec_2023", "kind": "survey_period", "value": 2023},
            {"id": "asec_2024", "kind": "survey_period", "value": 2024},
        ],
    }
    assert normative["stage_asset"] == {
        "id": "source_stages",
        "sha256": "788b815f748abb2061c41efb0cec4cc4952435dbcb4448ecf21ccac85a5ccec2",
    }
    assert operational["stage_asset"] == {
        "path": "microcosm.build.us/source_stages.json"
    }
    assert normative["stages"][0]["stage"] == "puf_tax_detail"
    assert normative["stages"][0]["artifacts"][0]["kind"] == ("public_microdata")

    operational_names = {
        "access",
        "documentation_url",
        "filename",
        "locator",
        "official_household_source",
        "official_person_source",
        "report_source",
        "repository",
        "repository_name_parts",
        "repository_owner",
        "revision",
    }
    leaked_bindings = [
        path
        for path, _ in _walk_values(normative)
        if path.rsplit("/", 1)[-1] in operational_names
        or path.rsplit("/", 1)[-1].endswith("_path_parts")
    ]
    leaked_urls = [
        path
        for path, value in _walk_values(normative)
        if isinstance(value, str) and "://" in value
    ]
    assert leaked_bindings == []
    assert leaked_urls == []
    assert any(
        isinstance(value, str) and "://" in value
        for _, value in _walk_values(operational)
    )


def test_source_schema_rejects_non_hex_content_digests() -> None:
    sources = build_sources()
    sources["sources"][0]["sha256"] = "x" * 64
    with pytest.raises(SpecValidationError, match="sha256"):
        load_schema_registry().validate(sources, "sources.schema.json")


def test_cd_vintage_crosswalk_source_and_geography_authority_are_pinned() -> None:
    crosswalk_source = next(
        row
        for row in build_sources()["sources"]
        if row["id"] == "us_congressional_district_vintage_crosswalk_117_to_119"
    )
    assert crosswalk_source == {
        "id": "us_congressional_district_vintage_crosswalk_117_to_119",
        "role": "congressional_district_vintage_crosswalk",
        "sha256": ("c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec"),
        "byte_size": 77_935,
        "loader": "kernel:load_congressional_district_vintage_crosswalk",
        "vintages": ["vintage:cd_117", "vintage:cd_119"],
        "vintage_authorities": [
            {
                "id": "cd_117",
                "kind": "geography_vintage",
                "value": "117th_congress",
            }
        ],
    }

    assignment = build_geography()["assignment"]
    assert assignment["order"] == "before_gap_fill"
    assert assignment["congressional_district_vintage_crosswalk"] == {
        "source_ref": ("source:us_congressional_district_vintage_crosswalk_117_to_119"),
        "source_vintage": "vintage:cd_117",
        "target_vintage": "vintage:cd_119",
    }
    load_schema_registry().validate(build_geography(), "geography.schema.json")


def test_cd_vintage_crosswalk_source_reference_is_resolved() -> None:
    resources = {**_vintage_resources(), "geography": build_geography()}
    _resolve_vintages(resources)

    resources["geography"]["assignment"]["congressional_district_vintage_crosswalk"][
        "source_ref"
    ] = "source:missing_crosswalk"
    with pytest.raises(SpecResolutionError, match="dangling source reference"):
        _resolve_vintages(resources)


def test_source_operational_locator_is_outside_spec_hash() -> None:
    sources = build_sources()
    original_hash, _, original_surfaces = _source_surface_hashes(sources)
    mutated_sources = copy.deepcopy(sources)
    mutated_sources["stages"][0]["source"] = "https://operational.invalid/moved"
    mutated_sources["stages"][0]["artifacts"][0]["locator"] = (
        "https://operational.invalid/rebound"
    )
    mutated_hash, _, mutated_surfaces = _source_surface_hashes(mutated_sources)

    assert mutated_hash == original_hash
    assert mutated_surfaces["operational"] != original_surfaces["operational"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda sources: sources["sources"][0].update({"sha256": "f" * 64}),
        lambda sources: sources["sources"][0].update(
            {"role": "reviewed_semantic_role_mutation"}
        ),
        lambda sources: sources["sources"][0].update(
            {"vintages": ["vintage:asec_2024"]}
        ),
    ],
)
def test_source_normative_pin_mutation_changes_spec_hash(
    mutation: Mutation,
) -> None:
    sources = build_sources()
    original_hash, _, _ = _source_surface_hashes(sources)
    mutated_sources = copy.deepcopy(sources)
    mutation(mutated_sources)
    mutated_hash, _, _ = _source_surface_hashes(mutated_sources)

    assert mutated_hash != original_hash


def test_source_prose_is_documentation_only() -> None:
    sources = build_sources()
    original_hash, original_docs_hash, _ = _source_surface_hashes(sources)
    mutated_sources = copy.deepcopy(sources)
    mutated_sources["stage_manifest"]["policy"] += " Documentation-only edit."
    mutated_hash, mutated_docs_hash, _ = _source_surface_hashes(mutated_sources)

    assert mutated_hash == original_hash
    assert mutated_docs_hash != original_docs_hash


def test_all_us_vintages_have_reciprocal_reviewed_compatibility() -> None:
    sources = build_sources()
    vintages = build_vintages()
    records = vintages["records"]
    assert len(records) == 16
    by_id = {record["id"]: record for record in records}
    assert set(by_id) == {
        "acs_2022",
        "acs_2024",
        "asec_2022",
        "asec_2023",
        "asec_2024",
        "cd_117",
        "cd_119",
        "census_2020",
        "org_2024",
        "policyengine_us_surface",
        "puma_2020",
        "release_us_2024",
        "scf_2022",
        "sipp_2023",
        "target_2024",
        "tax_2015",
    }
    for record_id, record in by_id.items():
        assert "value" not in record
        assert set(record) == {"id", "kind", "authority_ref", "compatible_with"}
        assert record["compatible_with"]
        for reference in record["compatible_with"]:
            target_id = reference.removeprefix("vintage:")
            assert f"vintage:{record_id}" in by_id[target_id]["compatible_with"]

    resolved = _resolve_vintages(
        {
            **_vintage_resources(),
            "sources": sources,
            "vintages": vintages,
        }
    )
    resolved_wire = thaw_json(resolved.vintage_authorities)
    assert len(resolved_wire["records"]) == 16
    assert (
        resolved_wire["records"]["policyengine_us_surface"]["value"]
        == (ENGINE_LOCK["engine"]["version"])
    )
    assert resolved_wire["engine_abi_lock_sha256"]


def test_release_target_and_engine_vintages_point_to_single_authorities() -> None:
    records = {row["id"]: row for row in build_vintages()["records"]}

    assert build_bundle()["dataset_run"] == {"target_period": 2024}
    assert records["target_2024"]["authority_ref"] == {
        "kind": "dataset_run",
        "pointer": "/dataset_run/target_period",
    }
    assert build_publication()["release"]["line"]["value"] == "microcosm-us-2024"
    assert records["release_us_2024"]["authority_ref"] == {
        "kind": "publication_release",
        "pointer": "/release/line/value",
    }
    assert records["policyengine_us_surface"]["authority_ref"] == {
        "kind": "engine_abi_lock",
        "pointer": "/engine/version",
    }


def test_vintage_schema_rejects_a_reintroduced_literal_authority() -> None:
    vintages = build_vintages()
    record = vintages["records"][0]
    del record["authority_ref"]
    record["value"] = 2022

    with pytest.raises(SpecValidationError, match="authority_ref"):
        load_schema_registry().validate(vintages, "vintages.schema.json")


def test_vintage_authority_value_mutation_changes_resolved_spec_identity() -> None:
    resources = _vintage_resources()
    before = _resolve_vintages(resources)
    before_binding = thaw_json(before.vintage_authorities)
    before_hash = sha256_json(
        spec_envelope(
            country="us",
            schema_version=1,
            normative_files={},
            resolved_bindings={"vintage_authorities": before_binding},
        )
    )

    mutated = copy.deepcopy(resources)
    source = next(
        row for row in mutated["sources"]["sources"] if row["id"] == "acs_rent_donor"
    )
    source["vintage_authorities"][0]["value"] = 2021
    after = _resolve_vintages(mutated)
    after_binding = thaw_json(after.vintage_authorities)
    after_hash = sha256_json(
        spec_envelope(
            country="us",
            schema_version=1,
            normative_files={},
            resolved_bindings={"vintage_authorities": after_binding},
        )
    )

    assert build_vintages() == resources["vintages"]
    assert before_binding["records"]["acs_2022"]["value"] == 2022
    assert after_binding["records"]["acs_2022"]["value"] == 2021
    assert after_hash != before_hash


def test_vintage_authority_requires_a_lowercase_hex_content_digest() -> None:
    resources = _vintage_resources()
    source = next(
        row for row in resources["sources"]["sources"] if row["id"] == "acs_rent_donor"
    )
    source["sha256"] = "g" * 64
    with pytest.raises(SpecResolutionError, match="must be content-pinned"):
        _resolve_vintages(resources)


def test_engine_lock_version_and_digest_are_spec_identity_bindings() -> None:
    resources = _vintage_resources()
    before = thaw_json(_resolve_vintages(resources).vintage_authorities)
    mutated_lock = copy.deepcopy(ENGINE_LOCK)
    mutated_lock["engine"]["version"] = "9.9.9"
    after = thaw_json(
        _resolve_vintages(resources, engine_lock=mutated_lock).vintage_authorities
    )

    assert (
        before["records"]["policyengine_us_surface"]["value"]
        != after["records"]["policyengine_us_surface"]["value"]
    )
    assert before["engine_abi_lock_sha256"] != after["engine_abi_lock_sha256"]
    assert sha256_json(before) != sha256_json(after)


def test_duplicate_normalized_vintage_authority_is_refused() -> None:
    resources = _vintage_resources()
    duplicate = copy.deepcopy(
        next(
            row
            for row in resources["sources"]["sources"]
            if row["id"] == "acs_rent_donor"
        )["vintage_authorities"][0]
    )
    resources["sources"]["sources"][0].setdefault("vintage_authorities", []).append(
        duplicate
    )

    with pytest.raises(
        SpecResolutionError,
        match="duplicate normalized vintage authority 'acs_2022'",
    ):
        _resolve_vintages(resources)


def test_dangling_or_kind_incompatible_vintage_authority_is_refused() -> None:
    resources = _vintage_resources()
    record = next(
        row for row in resources["vintages"]["records"] if row["id"] == "acs_2022"
    )
    record["authority_ref"]["authority"] = "missing"
    with pytest.raises(SpecResolutionError, match="dangling authority 'missing'"):
        _resolve_vintages(resources)

    resources = _vintage_resources()
    record = next(
        row for row in resources["vintages"]["records"] if row["id"] == "acs_2022"
    )
    record["kind"] = "tax_period_ref"
    with pytest.raises(SpecResolutionError, match="disagrees with source authority"):
        _resolve_vintages(resources)


def test_us_catalog_has_complete_explicit_contracts() -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    catalog = build_catalogs()
    columns = catalog["columns"]
    assert len(columns) == 176
    assert len({row["key"] for row in columns}) == 176
    assert catalog["metadata_waivers"] == [
        {
            "id": "policyengine_us_unit_unavailable",
            "field": "unit",
            "authority": "PolicyEngineUSVariableMetadataIndex",
            "public": False,
            "expires_on": "2026-11-16",
            "reason": (
                "The import-free installed-engine index exposes entity, dtype, "
                "and period but no physical unit. Numeric columns remain "
                "internal until a reviewed unit authority lands."
            ),
        }
    ]
    for row in columns:
        contract = row["contract"]
        assert row["key"].startswith(f"{contract['entity']}.")
        assert contract["domain"]
        assert contract["public_stability"] == "internal"
        if contract["unit"] == "unit_not_declared_by_engine_metadata":
            assert contract["unit_waiver"] == "policyengine_us_unit_unavailable"
        else:
            assert "unit_waiver" not in contract
    assert Counter(row["contract"]["unit"] for row in columns) == {
        "unit_not_declared_by_engine_metadata": 89,
        "boolean": 51,
        "count": 27,
        "categorical": 9,
    }
    assert {
        row["key"] for row in columns if row["key"].endswith(".@resolved_weight")
    } == {
        f"{entity}.@resolved_weight"
        for entity in (
            "family",
            "household",
            "marital_unit",
            "person",
            "spm_unit",
            "tax_unit",
        )
    }
    load_schema_registry().validate(catalog, "catalogs.schema.json")

    resolved = _resolve_vintages(
        {
            **_vintage_resources(),
            "catalogs": catalog,
        }
    )
    assert len(resolved.columns) == 176
    by_key = {row["key"]: row["contract"] for row in columns}
    for column in resolved.columns:
        contract = by_key[column.key]
        assert column.entity.id == contract["entity"]
        assert column.dtype == contract["dtype"]
        assert column.unit == contract["unit"]
        assert column.period == contract["definition_period"]
        assert column.vintage == contract["vintage"]
        assert column.nullable is contract["nullable"]
        assert column.domain == contract["domain"]
        assert column.public_stability == contract["public_stability"]
        assert column.unit_waiver == contract.get("unit_waiver")


@pytest.mark.parametrize("field", ["unit", "domain", "public_stability"])
def test_catalog_schema_requires_complete_contract_fields(field: str) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    catalog = build_catalogs()
    del catalog["columns"][0]["contract"][field]
    with pytest.raises(SpecValidationError, match=field):
        load_schema_registry().validate(catalog, "catalogs.schema.json")
