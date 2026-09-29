"""Tests split from packages/microcosm-build/tests/test_us_spec_bundle.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_spec_bundle import *


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
    __import__("policyengine_us")
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


def test_authored_imputation_sha256_fields_are_assets_or_policy_identity(
    generated_documents: dict[str, dict[str, object]],
) -> None:
    imputation = generated_documents["imputation.yaml"]
    graph = imputation["producer_graph"]
    assert "declared_attestations" not in graph
    assert "source_stage_asset" not in graph["resource_semantics"]

    asset_pins: list[tuple[str, str]] = []
    identity_digests: list[tuple[tuple[str, ...], str]] = []

    def collect_sha256(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*path, key)
                if key.endswith("sha256"):
                    if key == "asset_sha256":
                        asset_pins.append((str(value["asset"]), str(child)))
                    elif isinstance(child, dict) and "resolver_op" in child:
                        # A resolver binding names the runtime op that
                        # produces the digest; it is neither an authored
                        # digest nor an asset pin.
                        pass
                    else:
                        identity_digests.append((child_path, str(child)))
                collect_sha256(child, child_path)
        elif isinstance(value, list):
            for child in value:
                collect_sha256(child, (*path, "[]"))

    collect_sha256(imputation)
    assert identity_digests == [
        (
            (
                "models",
                "regime_gated_qrf",
                "post_draw_calibration",
                "sha256",
            ),
            str(
                imputation["models"]["regime_gated_qrf"]["post_draw_calibration"][
                    "sha256"
                ]
            ),
        ),
        (
            (
                "transfer_execution",
                "structural_target_policies",
                "is_pregnant",
                "sha256",
            ),
            str(
                imputation["transfer_execution"]["structural_target_policies"][
                    "is_pregnant"
                ]["sha256"]
            ),
        ),
    ]
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
    assert graph["schedule_payload_schema_version"] == 17
    assert graph["execution_receipt_contract"]["version"] == 4
    assert graph["execution_receipt_contract"]["transition_authority"]["version"] == 2
    assert graph["resource_semantics"]["schema_version"] == 2
    assert primary_binding["schema_version"] == 5
    assert primary_binding["qrf"]["worker_execution"] == {
        "surface": "execution_profile",
        "resolve_as": "worker_execution",
        "template": {
            "schema_version": 1,
            "semantic_identity": {
                "resolver_op": "primary_qrf_worker_semantic_identity"
            },
            "semantic_identity_sha256": {
                "resolver_op": "primary_qrf_worker_semantic_identity_sha256"
            },
            "audit_aliases": {"resolver_op": "primary_qrf_worker_audit_aliases"},
        },
    }
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
            if resource["binding"]["resource_kind"] == "primary_puf_execution_config"
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


def test_generated_bundle_reuses_primed_worker_identity(
    generated_documents: dict[str, dict[str, object]],
    request: pytest.FixtureRequest,
) -> None:
    """Generation and session priming must share one real worker attestation."""
    from microcosm.build.us_runtime import worker_identity

    assert "imputation.yaml" in generated_documents
    generated_identity = worker_identity.primary_qrf_worker_semantic_identity()
    request.getfixturevalue("_session_primary_qrf_worker_identities")
    assert worker_identity.primary_qrf_worker_semantic_identity() is generated_identity
