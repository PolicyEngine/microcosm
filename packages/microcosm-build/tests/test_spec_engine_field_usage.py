"""Exact-pointer and honest-effect tests for the F0 field-usage ledger."""

from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from microcosm.build.spec_engine.compiler_ir import compile_spec
from microcosm.build.spec_engine.field_usage import (
    EXPECTED_CONFIGURATION_FIELD_COUNT,
    FieldUsageError,
    Generation0Effect,
    UsageMode,
    build_field_usage_ledger,
    default_usage_claims,
)
from microcosm.build.spec_engine.legacy_adapter import (
    compile_to_legacy_payload,
    diff_legacy_payloads,
)
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import (
    ResolvedSpec,
    ResourceKind,
    freeze_json,
)
from microcosm.build.spec_engine.yaml12 import load_yaml12


@pytest.fixture(scope="module")
def resolved_us() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def compiled_us(resolved_us: ResolvedSpec):
    return compile_spec(resolved_us)


@pytest.fixture(scope="module")
def legacy_us(resolved_us: ResolvedSpec) -> dict[str, object]:
    return compile_to_legacy_payload(resolved_us)


@pytest.fixture(scope="module")
def field_ledger(resolved_us: ResolvedSpec, compiled_us, legacy_us):
    return build_field_usage_ledger(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _mutated_bundle(
    source: ResolvedSpec,
    destination: Path,
    kind: ResourceKind,
    mutation,
) -> ResolvedSpec:
    package_root = Path(__file__).parents[1] / "src/microcosm/build/us"
    mutated_root = shutil.copytree(
        package_root,
        destination / "mutated",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    relative = source.resource(kind).descriptor.path
    path = mutated_root / relative
    document = load_yaml12(path.read_text(encoding="utf-8"), source=str(path))
    assert isinstance(document, dict)
    mutation(document)
    path.write_text(
        yaml.safe_dump(
            document,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
            width=4096,
        ),
        encoding="utf-8",
    )
    return load_bundle(mutated_root)


def test_exact_complete_ledger_has_one_primary_mode_per_pointer(field_ledger) -> None:
    assert len(field_ledger.fields) == EXPECTED_CONFIGURATION_FIELD_COUNT == 42_419
    assert field_ledger.source_counts == {
        "authored": 32_403,
        "resolved_bindings": 10_016,
    }
    assert field_ledger.mode_counts == {
        "legacy_behavior": 13_983,
        "compiler_semantic": 27_985,
        "front_end_validation": 348,
        "identity_only": 103,
    }
    assert field_ledger.generation0_effect_counts == {
        "legacy_behavior": 38_456,
        "no_generation0_effect": 3_963,
    }
    assert len({field.pointer for field in field_ledger.fields}) == 42_419


def test_eligibility_concepts_are_validation_not_generation0_behavior(
    field_ledger,
) -> None:
    pointers = (
        "/authored/spec~1imputation.yaml/concepts/veteran_status/0",
        "/authored/spec~1imputation.yaml/waiver_records/0/code",
        "/authored/spec~1imputation.yaml/families/2/targets/0/requires_concepts/0",
        "/authored/spec~1imputation.yaml/families/2/targets/0/waiver",
    )
    for pointer in pointers:
        field = field_ledger.field(pointer)
        assert field.mode is UsageMode.FRONT_END_VALIDATION
        assert field.generation0_effect is Generation0Effect.NO_GENERATION0_EFFECT


def test_source_pins_channels_and_dtype_policy_are_honest_validation_routes(
    field_ledger,
) -> None:
    pointers = (
        "/authored/spec~1sources.yaml/sources/0/sha256",
        "/authored/spec~1spine.yaml/channels/0/observed_geography",
        "/authored/spec~1spine.yaml/assembly/shared_dtype_policy",
    )
    for pointer in pointers:
        field = field_ledger.field(pointer)
        assert field.mode is UsageMode.FRONT_END_VALIDATION
        assert field.generation0_effect is Generation0Effect.NO_GENERATION0_EFFECT


def test_stacked_geography_authority_fields_name_the_checkpoint_sink(
    field_ledger,
) -> None:
    pointers = (
        "/authored/spec~1geography.yaml/assignment/draw/asec/weight",
        "/authored/spec~1sources.yaml/sources/6/sha256",
        "/authored/spec~1sources.yaml/sources/7/vintage_authorities/0/value",
    )
    for pointer in pointers:
        field = field_ledger.field(pointer)
        assert field.mode is UsageMode.LEGACY_BEHAVIOR
        assert field.generation0_effect is Generation0Effect.LEGACY_BEHAVIOR
        assert (
            "/stacked_checkpoint_static_components/geography_assignment"
            in field.sink_pointers
        )


def test_spine_assembly_mass_share_fields_name_exact_adapter_sinks(
    field_ledger,
) -> None:
    for channel in ("acs", "asec"):
        field = field_ledger.field(
            "/authored/spec~1spine.yaml/assembly/household_mass_shares/"
            f"{channel}"
        )
        assert field.mode is UsageMode.LEGACY_BEHAVIOR
        assert field.generation0_effect is Generation0Effect.LEGACY_BEHAVIOR
        assert (
            f"/spine_assembly/household_mass_shares/{channel}"
            in field.sink_pointers
        )


def test_copied_surfaces_cannot_rescue_a_missing_calibration_sink(
    resolved_us: ResolvedSpec,
    compiled_us,
    legacy_us: dict[str, object],
) -> None:
    broken = deepcopy(legacy_us)
    calibration = _mapping(broken["calibration_contract"])
    solver = _mapping(calibration["solver"])
    stopping = _mapping(solver["stopping_contract"])
    del stopping["max_epochs"]

    with pytest.raises(
        FieldUsageError,
        match=r"calibration_contract/solver/stopping_contract/max_epochs",
    ):
        build_field_usage_ledger(
            resolved_us,
            compiled=compiled_us,
            legacy_payload=broken,
        )


def test_removing_identity_claim_leaves_selection_fields_unclaimed(
    resolved_us: ResolvedSpec,
    compiled_us,
    legacy_us: dict[str, object],
) -> None:
    claims = tuple(claim for claim in default_usage_claims() if claim.id != "selection")

    with pytest.raises(FieldUsageError, match=r"87 unclaimed normative field"):
        build_field_usage_ledger(
            resolved_us,
            compiled=compiled_us,
            legacy_payload=legacy_us,
            claims=claims,
        )


def test_duplicate_primary_claim_is_refused(
    resolved_us: ResolvedSpec,
    compiled_us,
    legacy_us: dict[str, object],
) -> None:
    claims = default_usage_claims()
    selection = next(claim for claim in claims if claim.id == "selection")
    duplicate = replace(selection, id="selection_duplicate")

    with pytest.raises(FieldUsageError, match=r"multiple primary claims"):
        build_field_usage_ledger(
            resolved_us,
            compiled=compiled_us,
            legacy_payload=legacy_us,
            claims=(*claims, duplicate),
        )


def test_corrupt_node_resolved_param_cannot_be_rescued_by_resource_copy(
    resolved_us: ResolvedSpec,
    compiled_us,
    legacy_us: dict[str, object],
) -> None:
    graph_node = compiled_us.producer_graph.nodes[0]
    node_index = compiled_us.producer_graph.order.index(graph_node.id)
    compiled_node = compiled_us.nodes[node_index]
    expected_path = f"/imputation/producer_graph/nodes/{graph_node.id}"
    param_index = next(
        index
        for index, param in enumerate(compiled_node.resolved_params)
        if param.path == expected_path
    )
    params = list(compiled_node.resolved_params)
    params[param_index] = replace(
        params[param_index],
        value=freeze_json({"corrupt": True}),
    )
    nodes = list(compiled_us.nodes)
    nodes[node_index] = replace(compiled_node, resolved_params=tuple(params))
    broken = replace(compiled_us, nodes=tuple(nodes))

    with pytest.raises(FieldUsageError, match=rf"node slice missing {graph_node.id}"):
        build_field_usage_ledger(
            resolved_us,
            compiled=broken,
            legacy_payload=legacy_us,
        )


def test_selection_mutation_remains_honestly_identity_only(
    tmp_path: Path,
    resolved_us: ResolvedSpec,
    legacy_us: dict[str, object],
) -> None:
    def mutate(document: dict[str, object]) -> None:
        algorithm = _mapping(document["algorithm"])
        design = _mapping(algorithm["design"])
        rejection = _mapping(design["rejection"])
        rejection["max_attempts"] = int(rejection["max_attempts"]) + 1

    mutated = _mutated_bundle(
        resolved_us,
        tmp_path,
        ResourceKind.SELECTION,
        mutate,
    )
    mutated_legacy = compile_to_legacy_payload(mutated)
    assert mutated.spec_sha256 != resolved_us.spec_sha256
    assert diff_legacy_payloads(legacy_us, mutated_legacy) == ()

    ledger = build_field_usage_ledger(mutated, legacy_payload=mutated_legacy)
    field = ledger.field(
        "/authored/spec~1selection.yaml/algorithm/design/rejection/max_attempts"
    )
    assert field.mode is UsageMode.IDENTITY_ONLY
    assert field.generation0_effect is Generation0Effect.NO_GENERATION0_EFFECT
    assert field.sink_pointers == ("/spec_binding/spec_sha256",)


def test_calibration_mutation_names_both_legacy_sinks(
    tmp_path: Path,
    resolved_us: ResolvedSpec,
    legacy_us: dict[str, object],
) -> None:
    def mutate(document: dict[str, object]) -> None:
        solver = _mapping(document["solver"])
        stopping = _mapping(solver["stopping_contract"])
        stopping["max_epochs"] = int(stopping["max_epochs"]) + 1

    mutated = _mutated_bundle(
        resolved_us,
        tmp_path,
        ResourceKind.CALIBRATION,
        mutate,
    )
    mutated_legacy = compile_to_legacy_payload(mutated)
    diff_paths = {
        difference.path
        for difference in diff_legacy_payloads(legacy_us, mutated_legacy)
    }
    assert {
        "/calibration_contract/solver/stopping_contract/max_epochs",
        "/calibration_contract/solver/stopping/max_epochs",
    } <= diff_paths

    ledger = build_field_usage_ledger(mutated, legacy_payload=mutated_legacy)
    field = ledger.field(
        "/authored/spec~1calibration.yaml/solver/stopping_contract/max_epochs"
    )
    assert field.mode is UsageMode.LEGACY_BEHAVIOR
    assert field.generation0_effect is Generation0Effect.LEGACY_BEHAVIOR
    assert {
        "/calibration_contract/solver/stopping_contract/max_epochs",
        "/calibration_contract/solver/stopping/max_epochs",
    } <= set(field.sink_pointers)


def test_mass_share_mutation_changes_the_named_adapter_surface(
    tmp_path: Path,
    resolved_us: ResolvedSpec,
    legacy_us: dict[str, object],
) -> None:
    def mutate(document: dict[str, object]) -> None:
        assembly = _mapping(document["assembly"])
        shares = _mapping(assembly["household_mass_shares"])
        shares["asec"] = 0.6
        shares["acs"] = 0.4

    mutated = _mutated_bundle(
        resolved_us,
        tmp_path,
        ResourceKind.SPINE,
        mutate,
    )
    mutated_legacy = compile_to_legacy_payload(mutated)
    diff_paths = {
        difference.path
        for difference in diff_legacy_payloads(legacy_us, mutated_legacy)
    }
    assert {
        "/spine_assembly/household_mass_shares/acs",
        "/spine_assembly/household_mass_shares/asec",
    } <= diff_paths

    ledger = build_field_usage_ledger(mutated, legacy_payload=mutated_legacy)
    for channel in ("acs", "asec"):
        field = ledger.field(
            "/authored/spec~1spine.yaml/assembly/household_mass_shares/"
            f"{channel}"
        )
        assert field.claim_id == "spine_assembly_household_mass_shares"
        assert (
            f"/spine_assembly/household_mass_shares/{channel}"
            in field.sink_pointers
        )


def test_geography_declaration_mutation_changes_checkpoint_identity(
    tmp_path: Path,
    resolved_us: ResolvedSpec,
    legacy_us: dict[str, object],
) -> None:
    def mutate(document: dict[str, object]) -> None:
        assignment = _mapping(document["assignment"])
        draw = _mapping(assignment["draw"])
        asec = _mapping(draw["asec"])
        asec["weight"] = "mutated_population_weight"

    mutated = _mutated_bundle(
        resolved_us,
        tmp_path,
        ResourceKind.GEOGRAPHY,
        mutate,
    )
    mutated_legacy = compile_to_legacy_payload(mutated)
    diff_paths = {
        difference.path
        for difference in diff_legacy_payloads(legacy_us, mutated_legacy)
    }
    pointer = "/authored/spec~1geography.yaml/assignment/draw/asec/weight"
    assert (
        "/stacked_checkpoint_static_components/geography_assignment/"
        "declaration/draw/asec/weight"
    ) in diff_paths
    field = build_field_usage_ledger(
        mutated,
        legacy_payload=mutated_legacy,
    ).field(pointer)
    assert field.claim_id == "geography_assignment"
    assert field.mode is UsageMode.LEGACY_BEHAVIOR


def test_geography_source_pin_mutation_changes_checkpoint_identity(
    tmp_path: Path,
    resolved_us: ResolvedSpec,
    legacy_us: dict[str, object],
) -> None:
    def mutate(document: dict[str, object]) -> None:
        sources = document["sources"]
        assert isinstance(sources, list)
        crosswalk = next(
            source
            for source in sources
            if isinstance(source, dict)
            and source.get("id")
            == "us_congressional_district_vintage_crosswalk_117_to_119"
        )
        crosswalk["sha256"] = "b" * 64

    mutated = _mutated_bundle(
        resolved_us,
        tmp_path,
        ResourceKind.SOURCES,
        mutate,
    )
    mutated_legacy = compile_to_legacy_payload(mutated)
    diff_paths = {
        difference.path
        for difference in diff_legacy_payloads(legacy_us, mutated_legacy)
    }
    assert (
        "/stacked_checkpoint_static_components/geography_assignment/authorities/"
        "congressional_district_vintage_crosswalk/sha256"
    ) in diff_paths
    field = build_field_usage_ledger(
        mutated,
        legacy_payload=mutated_legacy,
    ).field("/authored/spec~1sources.yaml/sources/7/sha256")
    assert field.claim_id == "source_geography_identity"
    assert field.mode is UsageMode.LEGACY_BEHAVIOR
