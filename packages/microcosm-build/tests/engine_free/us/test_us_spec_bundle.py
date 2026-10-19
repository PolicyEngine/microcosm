"""Tests split from packages/microcosm-build/tests/test_us_spec_bundle.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_spec_bundle import *


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
    assert len(resolved_country_spec.sources.stages) == 38
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

    assert len(sources["sources"]) == 8
    assert len(sources["stages"]) == 38

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
    assert (
        not {
            "edges",
            "input_inventories",
            "incomparable_node_policy",
            "order",
            "ordering",
            "transfer_groups",
            "waves",
        }
        & producer_graph.keys()
    )
    assert all(
        "depends_on" not in node and "write_scopes" not in node
        for node in producer_graph["nodes"]
    )
    primary_node = next(
        node for node in producer_graph["nodes"] if node["id"] == "primary_puf_qrf"
    )
    assert len(primary_node["outputs"]) == 35
    assert (
        sum(
            len(node["outputs"])
            for node in producer_graph["nodes"]
            if node["kind"] == "late_transfer"
        )
        == 0
    )
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
        == "e59c019d3d454eac99ac0ac209b6c5b6faaf9bdfcaeee18c36a25be19bf7da2f"
    )
    assert (
        compiled_schedule["payload_sha256"]
        == "7be038d34f228d66c12b53558fc5f30c93f1b376f1058c5e4fd7e7563a88d67f"
    )

    assert len(take_up["programs"]) == 17
    assert Counter(program["ownership"] for program in take_up["programs"]) == {
        "engine": 7,
        "measured": 1,
        "mixed": 1,
        "modeled": 7,
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
    assert len(take_up_steps) == 28
    assert Counter(step["kind"] for step in take_up_steps) == {
        "assignment": 5,
        "count_calibration": 5,
        "probability_seed": 5,
        "engine_default": 7,
        "measured_map": 3,
        "imputed_transfer": 2,
        "delivery_gate": 1,
    }
    source_backed_steps = [
        step for step in take_up_steps if "source_operation_ref" in step
    ]
    local_steps = [step for step in take_up_steps if "source_operation_ref" not in step]
    assert len(source_backed_steps) == 18
    assert len(local_steps) == 10
    assert (
        len({step["source_operation_ref"]["stage"] for step in source_backed_steps})
        == 9
    )
    source_operations = {
        stage["stage"]: stage["operations"] for stage in sources["stages"]
    }
    for step in source_backed_steps:
        assert "operation_id" not in step
        reference = step["source_operation_ref"]
        assert set(reference) == {"stage", "operation_index", "operation_id"}
        operation = source_operations[reference["stage"]][reference["operation_index"]]
        assert reference["operation_id"] == operation["kind"]
    assert all(
        isinstance(step["operation_id"], str) and step["operation_id"]
        for step in local_steps
    )
    assert all(step["kernel"].startswith("kernel:") for step in take_up_steps)
    assert len(battery["metric_registry"]) == 134
    assert len(battery["joint_metric_registry"]) == 1
    assert "metric_counts" not in battery
    assert "declared_surface" not in battery
    assert "completeness" not in battery
    battery_views = derive_battery_registry_views(battery)
    assert battery_views["metric_counts"] == {
        "boolean_incidence": 51,
        "categorical_tvd": 4,
        "monetary_sign_separated": 79,
    }
    assert set(calibration["targets"]) == {
        "cd_policy",
        "congressional_district",
        "county",
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
    assert len(catalogs["columns"]) == 176
    assert len(resolved_us_spec.columns) == 176
    assert Counter(artifact.kind for artifact in resolved_us_spec.artifacts) == {
        "producer_node": 38,
        "virtual_output": 18,
        "virtual_resource_binding": 28,
    }
    assert len(resolved_us_spec.scopes) == 7


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
    assert geography["assignment"]["order"] == "before_gap_fill"
    assert geography["assignment"]["assign_tract"] is False
    assert geography["assignment"]["congressional_district_vintage_crosswalk"] == {
        "source_ref": ("source:us_congressional_district_vintage_crosswalk_117_to_119"),
        "source_vintage": "vintage:cd_117",
        "target_vintage": "vintage:cd_119",
    }

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
    engine_version_occurrences = {
        kind: _count_scalar(_domain(resolved_us_spec, kind), engine_version)
        for kind in TYPED_DOMAIN_KINDS
    }
    assert engine_version_occurrences == {
        kind: int(kind == "take_up") for kind in TYPED_DOMAIN_KINDS
    }
    assert (
        take_up["legacy_contract_metadata"]["asserted_engine"][
            "inventory_built_against"
        ]
        == engine_version
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


def test_root_us_drafting_location_is_a_symlink_free_pointer() -> None:
    readme = AUTHORING_POINTER_ROOT / "README.md"
    assert readme.is_file()
    assert not readme.is_symlink()
    assert list(AUTHORING_POINTER_ROOT.rglob("*.yaml")) == []
    assert list(AUTHORING_POINTER_ROOT.rglob("*.yml")) == []
    assert "packages/microcosm-build/src/microcosm/build/us/spec" in (
        readme.read_text(encoding="utf-8")
    )
