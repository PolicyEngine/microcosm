"""Tests split from packages/microcosm-build/tests/test_spec_engine_inventory_coverage.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_inventory_coverage import *


def test_operational_free_digest_ignores_the_worker_execution_binding() -> None:
    """The primary-QRF worker binding authenticates a machine, not the spec.

    Its semantic identity hashes the interpreter binary, ABI, pyvenv.cfg, the
    installed distributions' RECORD files and the resolved CPU count, all of
    which differ between a macOS checkout and the Linux CI runners. The pinned
    inventory digests must not move with any of it, so the whole subtree is
    stripped before digesting, as are audit aliases and receipt self-hashes.
    """

    def receipt(worker_execution: object) -> dict[str, object]:
        binding: dict[str, object] = {"clone_attachment": {"seed": 991}}
        if worker_execution is not None:
            binding["worker_execution"] = worker_execution
        return {
            "producers": [
                {
                    "producer": "primary_puf_qrf",
                    "resources": {
                        "tax_unit.@primary_puf_execution_config": {
                            "binding": binding,
                            "audit_aliases": {"executable": "/some/venv/bin/python"},
                        }
                    },
                }
            ],
            "sha256": "self-hash over the unpruned receipt",
        }

    macos = receipt(
        {
            "schema_version": 1,
            "semantic_identity": {
                "interpreter": {"bytes_sha256": "a" * 64, "abi": {"soabi": "darwin"}},
                "installed_distributions_record_sha256": "b" * 64,
                "environment": {"POPULACE_FIT_PREDICT_WORKERS": {"resolved": 12}},
            },
            "semantic_identity_sha256": "c" * 64,
        }
    )
    linux = receipt(
        {
            "schema_version": 1,
            "semantic_identity": {
                "interpreter": {"bytes_sha256": "d" * 64, "abi": {"soabi": "linux"}},
                "installed_distributions_record_sha256": "e" * 64,
                "environment": {"POPULACE_FIT_PREDICT_WORKERS": {"resolved": 4}},
            },
            "semantic_identity_sha256": "f" * 64,
        }
    )
    unbound = receipt(None)

    digests = {
        inventory_coverage_module._operational_free_sha256(value)
        for value in (macos, linux, unbound)
    }

    assert len(digests) == 1
    stripped = inventory_coverage_module._without_operational_bindings(macos)
    resource = stripped["producers"][0]["resources"][
        "tax_unit.@primary_puf_execution_config"
    ]
    assert "worker_execution" not in resource["binding"]
    assert "audit_aliases" not in resource
    assert "sha256" not in stripped
    assert resource["binding"]["clone_attachment"] == {"seed": 991}


def test_nonexistent_bundle_home_is_rejected(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    corrupted = copy.deepcopy(report)
    item = corrupted["items"]["early_gap_fill_plan_exact"]
    item["bundle_homes"][0] = "/does/not/exist"

    with pytest.raises(InventoryCoverageError, match="bundle-home match evidence"):
        assert_inventory_coverage_complete(corrupted)


def test_bundle_home_wildcards_require_at_least_one_real_match() -> None:
    domains = {
        "imputation": {
            "nodes": [
                {"inputs": ["person.age"]},
                {"outputs": ["person.income"]},
            ]
        }
    }
    assert _bundle_home_matches(domains, "/imputation/nodes/*/inputs") == (
        ["person.age"],
    )
    assert _bundle_home_matches(domains, "/imputation/nodes/*/missing") == ()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda report: report.__setitem__("required_item_count", 0),
            "required_item_count differs",
        ),
        (
            lambda report: report.__setitem__("unexpected", True),
            "top-level fields differ",
        ),
        (
            lambda report: report["items"].pop("release_rungs_exact"),
            "item registry differs",
        ),
        (
            lambda report: report["counts"].__setitem__("producer_nodes", 0),
            "diagnostic counts differ",
        ),
        (
            _mark_inventory_item_missing,
            "inventory has missing required items",
        ),
    ],
)
def test_inventory_assertion_recomputes_tampered_summaries_and_items(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    corrupted = copy.deepcopy(report)
    mutation(corrupted)

    with pytest.raises(InventoryCoverageError, match=match):
        assert_inventory_coverage_complete(corrupted)


def test_operator_reorder_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    corrupted = copy.deepcopy(legacy_us)
    static = corrupted["stacked_checkpoint_static_components"]
    assert isinstance(static, dict)
    pool_code = static["pool_code"]
    assert isinstance(pool_code, dict)
    order = pool_code["post_clone_source_operator_order"]
    assert isinstance(order, list)
    order[0], order[1] = order[1], order[0]

    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=corrupted,
    )
    _assert_named_failure(report, "post_clone_operator_order_exact")


def test_ownership_cell_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    corrupted = copy.deepcopy(legacy_us)
    imputation = corrupted["imputation"]
    assert isinstance(imputation, dict)
    overlap = imputation["overlap_ownership"]
    assert isinstance(overlap, dict)
    ownership = overlap["ownership"]
    assert isinstance(ownership, list)
    first = ownership[0]
    assert isinstance(first, dict)
    first["final_owner"] = "corrupt:final_owner"

    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=corrupted,
    )
    _assert_named_failure(report, "conditional_ownership_matrix_exact")


def test_take_up_step_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    def mutation(value: dict[str, Any]) -> None:
        tanf = next(row for row in value["programs"] if row["id"] == "tanf")
        tanf["pipeline"][0]["rate"]["value"] = 0.220

    corrupted = _mutate_domain(resolved_us, ResourceKind.TAKE_UP, mutation)
    report = build_inventory_coverage(
        corrupted,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    _assert_named_failure(report, "take_up_pipeline_steps_exact")


def test_seed_owner_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    first = compiled_us.seed_stream_map.sites[0]
    corrupted_first = replace(first, owners=(*first.owners, first.owners[0]))
    corrupted_seed_map = replace(
        compiled_us.seed_stream_map,
        sites=(corrupted_first, *compiled_us.seed_stream_map.sites[1:]),
    )
    corrupted_ir = replace(compiled_us, seed_stream_map=corrupted_seed_map)

    report = build_inventory_coverage(
        resolved_us,
        compiled=corrupted_ir,
        legacy_payload=legacy_us,
    )
    _assert_named_failure(report, "seed_site_owner_bindings_exact")


def test_seed_site_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    first = compiled_us.seed_stream_map.sites[0]
    corrupted_first = replace(first, stream="build_model")
    corrupted_seed_map = replace(
        compiled_us.seed_stream_map,
        sites=(corrupted_first, *compiled_us.seed_stream_map.sites[1:]),
    )
    corrupted_ir = replace(compiled_us, seed_stream_map=corrupted_seed_map)

    report = build_inventory_coverage(
        resolved_us,
        compiled=corrupted_ir,
        legacy_payload=legacy_us,
    )
    _assert_named_failure(report, "seed_site_definitions_exact")
