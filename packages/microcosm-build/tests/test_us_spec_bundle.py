"""Generation-1 US bundle migration and package-seam gates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from microcosm.build.country_spec import ResolvedCountrySpec, load_country_spec
from microcosm.build.spec_engine import (
    F0_KERNEL_REGISTRY,
    ResolvedSpec,
    ResourceKind,
    SpecResolutionError,
    SpecValidationError,
    load_bundle,
    load_schema_registry,
)
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.spec_engine.resolver import resolve_cross_references
from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL
from microcosm.build.spec_engine.yaml12 import load_yaml12_file
from microcosm.build.us_runtime.take_up_contract import take_up_contract_identity
from tools.us_bundle_generation.contracts import derive_battery_registry_views
from tools.us_bundle_generation.core import (
    project_publication_legacy_release,
    project_spine_legacy_sampling,
)
from tools.us_bundle_generation.imputation import (
    derive_primary_effective_predictor_tuples,
    project_imputation_legacy_payloads,
)

ROOT = Path(__file__).resolve().parents[3]
US_PACKAGE_ROOT = ROOT / "packages/microcosm-build/src/microcosm/build/us"
US_SPEC_ROOT = US_PACKAGE_ROOT / "spec"
AUTHORING_POINTER_ROOT = ROOT / "specs/us"


def _load_generator_module():
    path = ROOT / "tools/generate_us_bundle_from_constants.py"
    spec = importlib.util.spec_from_file_location(
        "generate_us_bundle_from_constants",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_generator_module()

TYPED_DOMAIN_KINDS = frozenset(
    {
        "battery",
        "bundle",
        "calibration",
        "catalogs",
        "geography",
        "imputation",
        "publication",
        "selection",
        "sources",
        "spine",
        "take_up",
        "vintages",
    }
)

F_P_WAIVED_TARGETS = frozenset(
    {
        "has_champva_health_coverage_at_interview",
        "has_esi",
        "has_indian_health_service_coverage_at_interview",
        "has_marketplace_health_coverage_at_interview",
        "has_medicaid_health_coverage_at_interview",
        "has_non_marketplace_direct_purchase_health_coverage_at_interview",
        "has_other_means_tested_health_coverage_at_interview",
        "has_tricare_health_coverage_at_interview",
        "has_va_health_coverage_at_interview",
        "is_tanf_enrolled",
        "receives_housing_assistance",
        "receives_snap",
        "receives_wic",
        "takes_up_housing_assistance_if_eligible",
        "takes_up_medicare_if_eligible",
        "would_claim_wic",
    }
)

EXPECTED_F_P_CONCEPTS = {
    "american_indian_status": ["is_american_indian_or_alaska_native"],
    "citizenship_status": ["is_us_citizen"],
    "dependent_child_status": ["own_children_in_household"],
    "disability_status": ["has_hearing_difficulty", "has_vision_difficulty"],
    "employment_attachment": ["hours_worked_last_week", "weeks_worked_last_year"],
    "household_income_eligibility": ["spm_unit_net_income", "spm_unit_size"],
    "housing_need": ["pre_subsidy_rent", "tenure_type"],
    "medicare_coverage_context": ["acs_hins_medicare"],
    "military_coverage_context": ["acs_hins_va"],
    "pregnancy_status": ["is_pregnant"],
    "private_coverage_context": [
        "acs_hins_employer",
        "acs_hins_direct_purchase",
    ],
    "public_coverage_context": [
        "acs_hins_medicaid",
        "acs_hins_other_public",
    ],
    "veteran_status": ["is_veteran", "receives_va_payments"],
}

EXPECTED_F_P_TARGET_CONCEPTS = {
    "has_champva_health_coverage_at_interview": [
        "veteran_status",
        "military_coverage_context",
    ],
    "has_esi": ["employment_attachment", "private_coverage_context"],
    "has_indian_health_service_coverage_at_interview": ["american_indian_status"],
    "has_marketplace_health_coverage_at_interview": [
        "citizenship_status",
        "household_income_eligibility",
        "private_coverage_context",
    ],
    "has_medicaid_health_coverage_at_interview": [
        "dependent_child_status",
        "disability_status",
        "household_income_eligibility",
        "public_coverage_context",
    ],
    "has_non_marketplace_direct_purchase_health_coverage_at_interview": [
        "household_income_eligibility",
        "private_coverage_context",
    ],
    "has_other_means_tested_health_coverage_at_interview": [
        "disability_status",
        "household_income_eligibility",
        "public_coverage_context",
    ],
    "has_tricare_health_coverage_at_interview": ["veteran_status"],
    "has_va_health_coverage_at_interview": [
        "veteran_status",
        "military_coverage_context",
    ],
    "is_tanf_enrolled": [
        "dependent_child_status",
        "household_income_eligibility",
    ],
    "receives_housing_assistance": [
        "household_income_eligibility",
        "housing_need",
    ],
    "receives_snap": [
        "dependent_child_status",
        "disability_status",
        "household_income_eligibility",
    ],
    "receives_wic": [
        "dependent_child_status",
        "household_income_eligibility",
        "pregnancy_status",
    ],
    "takes_up_housing_assistance_if_eligible": [
        "household_income_eligibility",
        "housing_need",
    ],
    "takes_up_medicare_if_eligible": [
        "disability_status",
        "medicare_coverage_context",
    ],
    "would_claim_wic": ["dependent_child_status", "pregnancy_status"],
}

EXPECTED_RUNGS = [
    {"token": "f001", "fraction": 0.01, "percent_basis_points": 100},
    {"token": "f004", "fraction": 0.04, "percent_basis_points": 400},
    {"token": "f010", "fraction": 0.1, "percent_basis_points": 1_000},
    {"token": "f025", "fraction": 0.25, "percent_basis_points": 2_500},
    {"token": "f100", "fraction": 1.0, "percent_basis_points": 10_000},
]

LEGACY_COMPATIBILITY_SHA256 = {
    "source_stages.json": (
        "a3e9ca87f43d74b3d83320ca77559f28452036cf60dfc16bee10a22d4784f672"
    ),
    "support_spine.json": (
        "68f37dc6ae6e0cde7ebccb53f88dd4a800e63456f838fa214ff98d1db8d815be"
    ),
    "take_up_contract.json": (
        "5852e96582793313782d1c3edfc4cfdd0358a1a9cfd54bfea5844cbb09e89bd4"
    ),
}


@pytest.fixture(scope="module")
def resolved_us_spec() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def resolved_country_spec() -> ResolvedCountrySpec:
    return load_country_spec("us")


@pytest.fixture(scope="module")
def generated_documents() -> dict[str, dict[str, object]]:
    # The extractor is intentionally exercised once per module: this proves
    # the checked-in package remains a projection of the live constants while
    # keeping the comparatively expensive PolicyEngine ABI read bounded.
    pytest.importorskip(
        "policyengine_us",
        reason="live-engine oracle: the wheels gate's venv installs no engine",
    )
    return generator.build_documents()


def _domain(spec: ResolvedSpec, kind: ResourceKind | str) -> dict[str, object]:
    value = spec.domain(kind).to_wire()
    assert isinstance(value, dict)
    return value


def _json_resource(name: str) -> dict[str, object]:
    value = json.loads((US_PACKAGE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _count_scalar(value: object, expected: object) -> int:
    if isinstance(value, dict):
        return sum(_count_scalar(child, expected) for child in value.values())
    if isinstance(value, list):
        return sum(_count_scalar(child, expected) for child in value)
    return int(value == expected)


def test_us_package_has_twelve_typed_domains_and_loads_through_one_seam(
    resolved_us_spec: ResolvedSpec,
    resolved_country_spec: ResolvedCountrySpec,
) -> None:
    manifest = _json_resource("country_package.json")
    resources = manifest["resources"]
    assert isinstance(resources, list)
    typed_rows = [row for row in resources if row["kind"] != "legacy_json"]

    assert len(typed_rows) == 12
    assert {row["kind"] for row in typed_rows} == TYPED_DOMAIN_KINDS
    assert all(
        row
        == {
            "path": f"spec/{row['kind']}.yaml",
            "kind": row["kind"],
            "schema_id": f"{row['kind']}.schema.json",
        }
        for row in typed_rows
    )
    assert {
        resource.descriptor.kind.value
        for resource in resolved_us_spec.resources
        if resource.descriptor.kind is not ResourceKind.LEGACY_JSON
    } == TYPED_DOMAIN_KINDS

    assert resolved_us_spec.country == "us"
    assert resolved_us_spec.spec_binding.attestation == "mirror-attested"
    assert re.fullmatch(r"[0-9a-f]{64}", resolved_us_spec.spec_sha256)
    assert resolved_country_spec.resolved_spec is not None
    assert resolved_country_spec.resolved_spec.spec_sha256 == (
        resolved_us_spec.spec_sha256
    )
    assert resolved_country_spec.sources is not None
    assert len(resolved_country_spec.sources.stages) == 37
    assert resolved_country_spec.support_spine is not None
    assert len(resolved_country_spec.support_spine.support_spine.sources) == 2


def test_constant_derived_domain_counts_are_complete(
    resolved_us_spec: ResolvedSpec,
) -> None:
    sources = _domain(resolved_us_spec, ResourceKind.SOURCES)
    bundle = _domain(resolved_us_spec, ResourceKind.BUNDLE)
    spine = _domain(resolved_us_spec, ResourceKind.SPINE)
    imputation = _domain(resolved_us_spec, ResourceKind.IMPUTATION)
    take_up = _domain(resolved_us_spec, ResourceKind.TAKE_UP)
    battery = _domain(resolved_us_spec, ResourceKind.BATTERY)
    calibration = _domain(resolved_us_spec, ResourceKind.CALIBRATION)
    selection = _domain(resolved_us_spec, ResourceKind.SELECTION)
    catalogs = _domain(resolved_us_spec, ResourceKind.CATALOGS)

    assert len(sources["sources"]) == 7
    assert len(sources["stages"]) == 37

    families = imputation["families"]
    family_counts = Counter(family["stage"] for family in families)
    target_counts = Counter()
    for family in families:
        target_counts[family["stage"]] += len(family["targets"])
    assert family_counts == {
        "gap_fill_stacked_spine": 13,
        "primary_puf_qrf": 1,
        "late_producer_dag": 19,
    }
    assert target_counts == {
        "gap_fill_stacked_spine": 48,
        "primary_puf_qrf": 65,
        "late_producer_dag": 70,
    }
    assert "primary_effective_predictor_tuples" not in imputation["chaining"]
    assert len(derive_primary_effective_predictor_tuples(imputation)) == 65
    itemization_batches = [
        family for family in families if "puf_tax_itemization__batch_" in family["id"]
    ]
    assert [len(family["targets"]) for family in itemization_batches] == [8, 8, 8, 8, 5]
    modeled_output_scopes = {
        stage: Counter(
            target["output_coverage_scope"]
            for family in families
            if family["stage"] == stage
            for target in family["targets"]
        )
        for stage in ("primary_puf_qrf", "late_producer_dag")
    }
    assert modeled_output_scopes == {
        "primary_puf_qrf": {"puf_clone": 64, "whole_pool": 1},
        "late_producer_dag": {"whole_pool": 70},
    }
    assert all(
        "output_coverage_scope" not in target
        for family in families
        if family["stage"] == "gap_fill_stacked_spine"
        for target in family["targets"]
    )

    producer_graph = imputation["producer_graph"]
    assert len(producer_graph["nodes"]) == 38
    assert len(producer_graph["ownership_matrix"]) == 18
    assert not {
        "edges",
        "input_inventories",
        "incomparable_node_policy",
        "order",
        "ordering",
        "transfer_groups",
        "waves",
    } & producer_graph.keys()
    assert all(
        "depends_on" not in node and "write_scopes" not in node
        for node in producer_graph["nodes"]
    )
    primary_node = next(
        node for node in producer_graph["nodes"] if node["id"] == "primary_puf_qrf"
    )
    assert len(primary_node["outputs"]) == 35
    assert sum(
        len(node["outputs"])
        for node in producer_graph["nodes"]
        if node["kind"] == "late_transfer"
    ) == 0
    assert sum(len(node["outputs"]) for node in producer_graph["nodes"]) == 92
    compiled_schedule = project_imputation_legacy_payloads(
        imputation,
        sources_document=sources,
        spine_document=spine,
        bundle_document=bundle,
    )["late_producer_schedule_receipt"]
    assert len(compiled_schedule["edges"]) == 71
    assert len(compiled_schedule["waves"]) == 6
    assert (
        compiled_schedule["schedule_sha256"]
        == "b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5"
    )
    assert (
        compiled_schedule["payload_sha256"]
        == "7766f2e94476cceb93d9730a74afb2ca6fed836068053f96fa4141bcc2f6154e"
    )

    assert len(take_up["programs"]) == 13
    assert Counter(program["ownership"] for program in take_up["programs"]) == {
        "engine": 4,
        "measured": 1,
        "mixed": 1,
        "modeled": 6,
        "transferred": 1,
    }
    take_up_steps = []
    for program in take_up["programs"]:
        if "pipeline" in program:
            take_up_steps.extend(program["pipeline"])
        else:
            take_up_steps.extend(
                step for segment in program["segments"] for step in segment["pipeline"]
            )
    assert len(take_up_steps) == 24
    assert Counter(step["kind"] for step in take_up_steps) == {
        "assignment": 5,
        "count_calibration": 5,
        "probability_seed": 4,
        "engine_default": 4,
        "measured_map": 3,
        "imputed_transfer": 2,
        "delivery_gate": 1,
    }
    source_backed_steps = [
        step for step in take_up_steps if "source_operation_ref" in step
    ]
    local_steps = [
        step for step in take_up_steps if "source_operation_ref" not in step
    ]
    assert len(source_backed_steps) == 17
    assert len(local_steps) == 7
    assert (
        len(
            {
                step["source_operation_ref"]["stage"]
                for step in source_backed_steps
            }
        )
        == 8
    )
    source_operations = {
        stage["stage"]: stage["operations"] for stage in sources["stages"]
    }
    for step in source_backed_steps:
        assert "operation_id" not in step
        reference = step["source_operation_ref"]
        assert set(reference) == {"stage", "operation_index", "operation_id"}
        operation = source_operations[reference["stage"]][
            reference["operation_index"]
        ]
        assert reference["operation_id"] == operation["kind"]
    assert all(
        isinstance(step["operation_id"], str) and step["operation_id"]
        for step in local_steps
    )
    assert all(step["kernel"].startswith("kernel:") for step in take_up_steps)
    assert len(battery["metric_registry"]) == 131
    assert len(battery["joint_metric_registry"]) == 1
    assert "metric_counts" not in battery
    assert "declared_surface" not in battery
    assert "completeness" not in battery
    battery_views = derive_battery_registry_views(battery)
    assert battery_views["metric_counts"] == {
        "boolean_incidence": 48,
        "categorical_tvd": 4,
        "monetary_sign_separated": 79,
    }
    assert set(calibration["targets"]) == {
        "cd_policy",
        "congressional_district",
        "county",
        "default_geography_layers",
        "facts_sha256",
        "geography_layers",
        "manifest_sha256",
        "matrix",
        "negative_target_policy",
        "source",
        "zero_target_policy",
    }
    assert calibration["targets"]["source"] == "chronicle_facts"
    assert len(calibration["tail_contracts"]) == 2
    assert len(calibration["refit_contracts"]) == 2
    assert isinstance(calibration["solver"]["initialization_contract"], dict)
    assert isinstance(calibration["solver"]["infeasibility_contract"], dict)
    assert isinstance(calibration["solver"]["target_priority_contract"], dict)
    assert set(calibration["solver"]["loss"]) == {"formula_id", "params"}
    for knob in ("k", "pi_hi", "seed"):
        assert selection["exact_k"][knob]["required"] is True
        assert selection["exact_k"][knob]["default"] is None
    assert len(catalogs["columns"]) == 173
    assert len(resolved_us_spec.columns) == 173
    assert Counter(artifact.kind for artifact in resolved_us_spec.artifacts) == {
        "producer_node": 38,
        "virtual_output": 18,
        "virtual_resource_binding": 28,
    }
    assert len(resolved_us_spec.scopes) == 7


def test_eligibility_blind_targets_have_one_explicit_f_p_waiver(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    imputation = generated_documents["imputation.yaml"]
    load_schema_registry().validate(imputation, "imputation.schema.json")
    assert imputation["concepts"] == EXPECTED_F_P_CONCEPTS
    assert len(imputation["waiver_records"]) == 1
    waiver = imputation["waiver_records"][0]
    assert waiver == {
        "id": "f_p_eligibility_concepts_absent",
        "code": "F-P",
        "marker": "F-P: eligibility concepts absent",
        "reason": "eligibility_concepts_absent",
        "coverage_status": ("required_concepts_not_covered_by_current_predictors"),
        "targets": sorted(F_P_WAIVED_TARGETS),
        "requires_concepts": sorted(EXPECTED_F_P_CONCEPTS),
        "missing_concepts_by_target": EXPECTED_F_P_TARGET_CONCEPTS,
    }

    targets_by_name: dict[str, list[dict[str, object]]] = {}
    for target in (
        target for family in imputation["families"] for target in family["targets"]
    ):
        targets_by_name.setdefault(target["name"], []).append(target)
    assert F_P_WAIVED_TARGETS <= targets_by_name.keys()
    for name, required in EXPECTED_F_P_TARGET_CONCEPTS.items():
        for target in targets_by_name[name]:
            assert target["requires_concepts"] == required
            assert target["waiver"] == waiver["id"]

    header = (US_SPEC_ROOT / "imputation.yaml").read_text(encoding="utf-8")
    assert "# F-P: eligibility concepts absent" in "\n".join(header.splitlines()[:3])


def _generated_resolution_resources(
    generated_documents: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        filename.removesuffix(".yaml"): copy.deepcopy(document)
        for filename, document in generated_documents.items()
    }


def _resolve_generated_resources(resources: dict[str, object]) -> None:
    resolve_cross_references(
        resources,
        kernel_registry=F0_KERNEL_REGISTRY,
        generated_authorities={
            "engine_abi_lock": _json_resource("engine_abi.lock.json")
        },
    )


def _participation_target(
    imputation: dict[str, object], target_name: str
) -> dict[str, object]:
    return next(
        target
        for family in imputation["families"]
        for target in family["targets"]
        if target["name"] == target_name
    )


def test_concept_coverage_load_phase_requires_a_target_listed_exact_waiver(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    resources = _generated_resolution_resources(generated_documents)
    imputation = resources["imputation"]
    target = _participation_target(
        imputation, "has_champva_health_coverage_at_interview"
    )
    target.pop("waiver")

    with pytest.raises(
        SpecResolutionError,
        match="valid target-listed F-P waiver is required",
    ):
        _resolve_generated_resources(resources)

    resources = _generated_resolution_resources(generated_documents)
    imputation = resources["imputation"]
    waiver = imputation["waiver_records"][0]
    waiver["missing_concepts_by_target"][
        "has_champva_health_coverage_at_interview"
    ].remove("military_coverage_context")
    with pytest.raises(
        SpecResolutionError,
        match="does not exactly record missing concepts",
    ):
        _resolve_generated_resources(resources)


def test_concept_coverage_requires_every_column_and_waiver_self_expires(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    resources = _generated_resolution_resources(generated_documents)
    imputation = resources["imputation"]
    champva_families = [
        family
        for family in imputation["families"]
        if any(
            target["name"] == "has_champva_health_coverage_at_interview"
            for target in family["targets"]
        )
    ]
    champva_block_ids = {
        block_id for family in champva_families for block_id in family["predictors"]
    }
    for block_id in champva_block_ids:
        imputation["predictor_blocks"][block_id]["columns"].append("is_veteran")
    # veteran_status still lacks receives_va_payments: partial coverage is not
    # coverage, so the exact waiver remains valid.
    _resolve_generated_resources(resources)

    for block_id in champva_block_ids:
        imputation["predictor_blocks"][block_id]["columns"].append(
            "receives_va_payments"
        )
    with pytest.raises(
        SpecResolutionError,
        match="does not exactly record missing concepts",
    ):
        _resolve_generated_resources(resources)


def test_f_p_schema_refuses_an_empty_waiver_concept_inventory(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    imputation = copy.deepcopy(generated_documents["imputation.yaml"])
    imputation["waiver_records"][0]["requires_concepts"] = []
    with pytest.raises(SpecValidationError, match="requires_concepts"):
        load_schema_registry().validate(imputation, "imputation.schema.json")


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("battery", ("gates", 0)),
        ("calibration", ("solver",)),
        ("imputation", ("families", 0)),
        ("selection", ("exact_k",)),
        ("sources", ("sources", 0)),
        ("spine", ("assembly",)),
        ("take_up", ("programs", 0)),
    ],
)
def test_nested_normative_objects_refuse_unknown_fields(
    generated_documents: dict[str, dict[str, object]],
    kind: str,
    path: tuple[str | int, ...],
) -> None:
    document = copy.deepcopy(generated_documents[f"{kind}.yaml"])
    nested: object = document
    for part in path:
        if isinstance(part, int):
            assert isinstance(nested, list)
            nested = nested[part]
        else:
            assert isinstance(nested, dict)
            nested = nested[part]
    assert isinstance(nested, dict)
    nested["unexpected_normative_field"] = True

    with pytest.raises(SpecValidationError, match="Additional properties"):
        load_schema_registry().validate(document, f"{kind}.schema.json")


def test_typed_domains_are_exact_legacy_compatibility_projections(
    resolved_us_spec: ResolvedSpec,
) -> None:
    documents = {
        f"{kind}.yaml": _domain(resolved_us_spec, kind) for kind in TYPED_DOMAIN_KINDS
    }
    projections = generator.legacy_compatibility_projections(documents)
    for name, projection in projections.items():
        assert projection == _json_resource(name)
    assert (
        take_up_contract_identity()["resource_sha256"]
        == hashlib.sha256(
            json.dumps(
                projections["take_up_contract.json"],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert {
        name: hashlib.sha256((US_PACKAGE_ROOT / name).read_bytes()).hexdigest()
        for name in LEGACY_COMPATIBILITY_SHA256
    } == LEGACY_COMPATIBILITY_SHA256


def test_typed_imputation_reconstructs_all_constants_authority_components(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    payloads = project_imputation_legacy_payloads(
        generated_documents["imputation.yaml"],
        sources_document=generated_documents["sources.yaml"],
        spine_document=generated_documents["spine.yaml"],
        bundle_document=generated_documents["bundle.yaml"],
    )

    assert set(payloads) == {
        "gap_fill_plan",
        "gap_fill_producer_schedule_receipt",
        "late_producer_resource_semantics",
        "late_producer_schedule_receipt",
        "overlap_ownership",
        "primary_qrf",
        "transfer_execution_contract_identities",
    }
    assert payloads["late_producer_resource_semantics"]["producer_count"] == 38
    assert len(payloads["overlap_ownership"]["ownership"]) == 18
    assert (
        payloads["overlap_ownership"]["sha256"]
        == "5f64f0aac49e2313177564f71876bffc8c81b3ded4df701e70930e60e9c98356"
    )


def test_authored_imputation_contains_only_external_asset_sha256_pins(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    imputation = generated_documents["imputation.yaml"]
    graph = imputation["producer_graph"]
    assert "declared_attestations" not in graph
    assert "source_stage_asset" not in graph["resource_semantics"]

    asset_pins: list[tuple[str, str]] = []

    def collect_sha256(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("sha256"):
                    assert key == "asset_sha256"
                    asset_pins.append((str(value["asset"]), str(child)))
                collect_sha256(child)
        elif isinstance(value, list):
            for child in value:
                collect_sha256(child)

    collect_sha256(imputation)
    assert len(asset_pins) == 2
    assert set(asset_pins) == {
        (
            "microcosm.build.us/soca_capital_gain_distribution_shares.json",
            "e7d31a5956dc420940002b5c5120b4fa7f6af6fa8bf071d488affe35b616611d",
        ),
        (
            "microcosm.build.us/soi_table_2_1_interest_components_ty2015.json",
            "c3356ae216487f365cb0e0a7ab1ba46843a6c52950b75eae1bfab9b0b80a735a",
        ),
    }

    late_target_derivations = [
        resource["dynamic_field"]["derivation"]
        for node in graph["nodes"]
        for resource in node["virtual_resources"]
        if resource["binding"]["resource_kind"] == "late_transfer_target_bank"
    ]
    assert len(late_target_derivations) == 19
    assert all(
        set(derivation) == {"base", "producer"}
        for derivation in late_target_derivations
    )

    primary_node = next(
        node for node in graph["nodes"] if node["id"] == "primary_puf_qrf"
    )
    primary_binding = next(
        resource["binding"]
        for resource in primary_node["virtual_resources"]
        if resource["binding"]["resource_kind"] == "primary_puf_execution_config"
    )
    runtime_bands = primary_binding["capital_gains_tail"]["soi_e19200_agi_bands"][
        "runtime_agi_bands"
    ]
    assert set(runtime_bands) == {"schema_version", "agi_bands"}


@pytest.mark.parametrize("asset_pin", ["share_asset", "soi_agi_bands"])
def test_imputation_schema_refuses_malformed_external_asset_sha256(
    generated_documents: dict[str, dict[str, object]],
    asset_pin: str,
) -> None:
    imputation = copy.deepcopy(generated_documents["imputation.yaml"])
    if asset_pin == "share_asset":
        share_asset = imputation["transfer_execution"]["post_transfer_features"][
            "schedule_d_capital_gain_distributions"
        ]["enabled_overrides"]["share_asset"]
        share_asset["asset_sha256"] = "not-a-sha256"
    else:
        primary_node = next(
            node
            for node in imputation["producer_graph"]["nodes"]
            if node["id"] == "primary_puf_qrf"
        )
        primary_binding = next(
            resource["binding"]
            for resource in primary_node["virtual_resources"]
            if resource["binding"]["resource_kind"]
            == "primary_puf_execution_config"
        )
        primary_binding["capital_gains_tail"]["soi_e19200_agi_bands"][
            "asset_sha256"
        ] = "not-a-sha256"

    with pytest.raises(SpecValidationError, match="asset_sha256"):
        load_schema_registry().validate(imputation, "imputation.schema.json")


@pytest.mark.parametrize("stage", ["primary_puf_qrf", "late_producer_dag"])
def test_imputation_schema_requires_modeled_target_output_scope(
    generated_documents: dict[str, dict[str, object]],
    stage: str,
) -> None:
    imputation = copy.deepcopy(generated_documents["imputation.yaml"])
    family = next(row for row in imputation["families"] if row["stage"] == stage)
    family["targets"][0].pop("output_coverage_scope")

    with pytest.raises(SpecValidationError, match="output_coverage_scope"):
        load_schema_registry().validate(imputation, "imputation.schema.json")


def test_imputation_schema_refuses_output_scope_on_gap_fill_target(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    imputation = copy.deepcopy(generated_documents["imputation.yaml"])
    family = next(
        row
        for row in imputation["families"]
        if row["stage"] == "gap_fill_stacked_spine"
    )
    family["targets"][0]["output_coverage_scope"] = "whole_pool"

    with pytest.raises(SpecValidationError, match="output_coverage_scope"):
        load_schema_registry().validate(imputation, "imputation.schema.json")


def test_imputation_projector_derives_hashes_and_joins_source_asset(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    imputation = generated_documents["imputation.yaml"]
    sources = generated_documents["sources.yaml"]
    baseline = project_imputation_legacy_payloads(
        imputation,
        sources_document=sources,
        spine_document=generated_documents["spine.yaml"],
        bundle_document=generated_documents["bundle.yaml"],
    )

    mutated_imputation = copy.deepcopy(imputation)
    mutated_imputation["producer_graph"]["graph_schema_version"] += 1
    mutated = project_imputation_legacy_payloads(
        mutated_imputation,
        sources_document=sources,
        spine_document=generated_documents["spine.yaml"],
        bundle_document=generated_documents["bundle.yaml"],
    )
    baseline_schedule = baseline["late_producer_schedule_receipt"]
    mutated_schedule = mutated["late_producer_schedule_receipt"]
    assert mutated_schedule["schedule_sha256"] != baseline_schedule["schedule_sha256"]
    assert mutated_schedule["payload_sha256"] != baseline_schedule["payload_sha256"]
    for producer in mutated["late_producer_resource_semantics"]["producers"]:
        for resource in producer["resources"].values():
            if resource["binding"]["resource_kind"] != "late_transfer_target_bank":
                continue
            derivation = resource["dynamic_field"]["derivation"]
            assert (
                derivation["late_producer_dag_sha256"]
                == (mutated_schedule["schedule_sha256"])
            )
            assert (
                derivation["late_producer_schedule_sha256"]
                == (mutated_schedule["payload_sha256"])
            )

    mutated_sources = copy.deepcopy(sources)
    mutated_sources["stage_asset"]["path"] = "source-owned/rebound.json"
    mutated_sources["stage_asset"]["sha256"] = "f" * 64
    source_rebound = project_imputation_legacy_payloads(
        imputation,
        sources_document=mutated_sources,
        spine_document=generated_documents["spine.yaml"],
        bundle_document=generated_documents["bundle.yaml"],
    )
    stage_specs = [
        resource["binding"]["source_stage_spec"]
        for producer in source_rebound["late_producer_resource_semantics"]["producers"]
        for resource in producer["resources"].values()
        if resource["binding"]["resource_kind"] == "post_clone_source_execution_config"
        and resource["binding"]["source_stage_spec"] is not None
    ]
    assert len(stage_specs) == 15
    assert {
        (stage_spec["asset"], stage_spec["asset_sha256"]) for stage_spec in stage_specs
    } == {("source-owned/rebound.json", "f" * 64)}


def test_legacy_seed_vintage_and_publication_grammars_are_pinned(
    resolved_us_spec: ResolvedSpec,
) -> None:
    bundle = _domain(resolved_us_spec, ResourceKind.BUNDLE)
    geography = _domain(resolved_us_spec, ResourceKind.GEOGRAPHY)
    publication = _domain(resolved_us_spec, ResourceKind.PUBLICATION)
    spine = _domain(resolved_us_spec, ResourceKind.SPINE)
    take_up = _domain(resolved_us_spec, ResourceKind.TAKE_UP)
    vintages = _domain(resolved_us_spec, ResourceKind.VINTAGES)

    assert bundle == {
        "country": "us",
        "dataset_run": {"target_period": 2024},
        "identity_generation": 1,
        "seed_protocol": LEGACY_V1_PROTOCOL.id,
    }
    assert len(LEGACY_V1_PROTOCOL.sites) == 53
    assert len(LEGACY_V1_PROTOCOL.streams) == 14
    assert LEGACY_V1_PROTOCOL.site("survey_sample_asec").default == 578
    assert LEGACY_V1_PROTOCOL.site("puf_live_aggregate_disaggregation").default == 0
    assert (
        LEGACY_V1_PROTOCOL.site("puf_archived_aggregate_disaggregation").default == 42
    )
    assert (
        LEGACY_V1_PROTOCOL.site("ssi_weighted_replacement_training").default
        == 8_386_123_572_872_638_692
    )
    assert (
        LEGACY_V1_PROTOCOL.site("sipp_tip_training_cap").default
        == 5_559_651_045_748_063_828
    )
    assert geography["phase"] == "legacy"
    assert geography["assignment"]["anchor"] == "puma"
    assert geography["assignment"]["order"] == "legacy_post_transfer"
    assert geography["assignment"]["assign_tract"] is False

    engine_pins = [
        record
        for record in vintages["records"]
        if record["kind"] == "policy_engine_surface_ref"
    ]
    assert engine_pins == [
        {
            "authority_ref": {
                "kind": "engine_abi_lock",
                "pointer": "/engine/version",
            },
            "compatible_with": ["vintage:target_2024"],
            "id": "policyengine_us_surface",
            "kind": "policy_engine_surface_ref",
        }
    ]
    engine_lock = _json_resource("engine_abi.lock.json")
    engine_version = engine_lock["engine"]["version"]
    assert (
        sum(
            _count_scalar(_domain(resolved_us_spec, kind), engine_version)
            for kind in TYPED_DOMAIN_KINDS
        )
        == 0
    )
    assert _count_scalar(engine_lock, engine_version) == 1
    resolved_vintages = thaw_json(resolved_us_spec.vintage_authorities)
    assert resolved_vintages["records"]["policyengine_us_surface"]["value"] == (
        engine_version
    )
    assert (
        resolved_vintages["engine_abi_lock_sha256"]
        == hashlib.sha256(
            (US_PACKAGE_ROOT / "engine_abi.lock.json").read_bytes()
        ).hexdigest()
    )
    assert "engine_abi" not in take_up

    release_series = next(
        record for record in vintages["records"] if record["id"] == "release_us_2024"
    )
    release = publication["release"]
    assert release_series["authority_ref"] == {
        "kind": "publication_release",
        "pointer": "/release/line/value",
    }
    assert "value" not in release_series
    assert release["line"]["value"] == "microcosm-us-2024"
    assert (
        sum(
            _count_scalar(_domain(resolved_us_spec, kind), "microcosm-us-2024")
            for kind in TYPED_DOMAIN_KINDS
        )
        == 1
    )
    assert release["line"]["normative"] is True
    target_period = next(
        record for record in vintages["records"] if record["id"] == "target_2024"
    )
    assert target_period["authority_ref"] == {
        "kind": "dataset_run",
        "pointer": "/dataset_run/target_period",
    }
    assert "value" not in target_period
    assert release["rung_fractions"] == EXPECTED_RUNGS
    assert "rungs" not in release
    assert "compiled_regex" not in release
    assert "legacy_compiled_regexes" not in release
    fraction = spine["sampling"]["fraction"]
    assert "rungs" not in fraction
    assert fraction["rungs_ref"] == {
        "domain": "publication",
        "pointer": "/release/rung_fractions",
    }
    projected_release = project_publication_legacy_release(publication)
    assert projected_release["rungs"] == [row["token"] for row in EXPECTED_RUNGS]
    assert (
        project_spine_legacy_sampling(spine, publication=publication)["fraction"][
            "rungs"
        ]
        == EXPECTED_RUNGS
    )

    compiled = re.compile(projected_release["compiled_regex"])
    assert compiled.fullmatch(
        "microcosm-us-2024-stacked-f025-s578-asec100-acs200-20260816T120000Z-deadbeef"
    )
    assert not compiled.fullmatch(
        "populace-us-2024-stacked-f025-s578-asec100-acs200-20260816T120000Z-deadbeef"
    )
    assert not compiled.fullmatch(
        "microcosm-us-2024-stacked-f050-s578-asec100-acs200-20260816T120000Z-deadbeef"
    )


def test_generator_reproduces_every_checked_in_bundle_byte(
    generated_documents: dict[str, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(generated_documents) == {f"{kind}.yaml" for kind in TYPED_DOMAIN_KINDS}
    for filename, expected in generated_documents.items():
        path = US_SPEC_ROOT / filename
        assert load_yaml12_file(path) == expected
        assert path.read_bytes() == generator.render_yaml(filename, expected)

    assert _json_resource("country_package.json") == generator.country_manifest()
    monkeypatch.setattr(
        generator,
        "build_documents",
        lambda: generated_documents,
    )
    assert generator.write_generated_files(check=True) == ()


def test_root_us_drafting_location_is_a_symlink_free_pointer() -> None:
    readme = AUTHORING_POINTER_ROOT / "README.md"
    assert readme.is_file()
    assert not readme.is_symlink()
    assert list(AUTHORING_POINTER_ROOT.rglob("*.yaml")) == []
    assert list(AUTHORING_POINTER_ROOT.rglob("*.yml")) == []
    assert "packages/microcosm-build/src/microcosm/build/us/spec" in (
        readme.read_text(encoding="utf-8")
    )
