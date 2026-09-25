"""Tests split from packages/microcosm-build/tests/test_uk_graph.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_graph import *


def test_uk_expand_contract_carries_cells_links_weights_and_design_lineage() -> None:
    expanded = patch(_expand_population(), _expand_node(), _expand_result())

    person = expanded.frame.table("person")
    assert person["person_id"].tolist() == [1, 2, 3]
    assert person["person_household_id"].tolist() == [10, 20, 30]
    assert person["hidden_payload"].tolist() == [1.25, 9.5, 1.25]
    assert expanded.frame.table("household")["is_clone"].tolist() == [
        False,
        False,
        True,
    ]
    assert expanded.frame.weights_for("household").kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(
        expanded.frame.weights_for("household").values,
        np.array([0.5, 2.0, 0.5]),
    )
    np.testing.assert_array_equal(
        expanded.design_weights["household"], np.array([1.0, 2.0, 1.0])
    )
    assert expanded.frame.mass_log[-1].reason == "test clone mass is conserved"
    assert expanded.mass_ledger[-1].operation == "expand"


def test_uk_expand_contract_rejects_unknown_source_ids() -> None:
    with pytest.raises(PopulationError, match="unknown 'person' source ids"):
        patch(_expand_population(), _expand_node(), _expand_result(bad_source=True))


def test_uk_spine_graph_contains_manifest_stages_and_named_exclusions() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    expected = tuple(
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    )
    graph = uk_spine_graph(spec)
    ids = {node.id for node in graph.nodes}

    # 32 with the #832 uc_reporter_redraw, #685 uc_deduction_attributes,
    # #791 frs_relationships, #725 hmrc_cgt_asset_type_spine, #970
    # cgt_incidence_anchor, #930 nts_bus_travel and the income-anchor lane's
    # spi_income_band_donors stages; the two named exclusions are the
    # certified-pair alternatives, not steps of this pipeline.
    assert len(expected) == 33
    assert UK_SPINE_EXCLUSIONS == {
        "frs_hmrc_retained_leaves",
        "hmrc_spi_income",
    }
    assert set(expected) <= ids
    assert not (UK_SPINE_EXCLUSIONS & ids)
    root_dtypes = {
        (owned.entity, owned.column): owned.dtype
        for owned in graph.node("create_uk_frs").outputs
    }
    assert root_dtypes[("person", "is_uc_claimant")] == "bool"
    assert {
        node.id for node in graph.nodes if node.structural is StructuralDelta.EXPAND
    } == UK_SPINE_STRUCTURAL_STAGES


def test_uk_spine_compile_order_is_derived_from_declared_inputs() -> None:
    spec = load_country_spec("uk")
    assert spec.sources is not None
    expected = tuple(
        stage.stage
        for stage in spec.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    )
    compiled = compile_graph(uk_spine_graph(spec))
    stage_order = tuple(node_id for node_id in compiled.order if node_id in expected)

    assert set(stage_order) == set(expected)
    assert len(stage_order) == len(expected)
    assert all(
        set(compiled.predecessors[node_id]) <= set(compiled.order[:index])
        for index, node_id in enumerate(compiled.order)
    )
    assert all(
        compiled.graph.node(node_id).inputs
        for node_id in expected[1:]
        if node_id not in UK_SPINE_STRUCTURAL_STAGES
    )
    graph = compiled.graph
    reversed_declaration = Graph(
        graph.country, graph.sources, tuple(reversed(graph.nodes))
    )
    assert compile_graph(reversed_declaration).order == compiled.order
    assert "frs_employment" in compiled.predecessors["frs_legacy_proxies"]
    assert "was_wealth" in compiled.predecessors["regional_property_uprating.boundary"]
    assert "regional_property_uprating" in compiled.predecessors["lcfs_consumption"]
    assert "spi_support_channel" in compiled.predecessors["spi_income_band_donors"]
    assert "spi_income_band_donors" in compiled.predecessors["hmrc_spi_income_spine"]


def test_uk_production_graph_binds_split_donor_sources_and_runtime_config() -> None:
    graph = uk_spine_graph(
        source_mode="split",
        sample_fraction=0.1,
        sample_seed=999,
    )

    assert {source.name for source in graph.sources} == {
        "frs",
        "was",
        "nts_household",
        "nts_individual",
        "nts_trip",
        "nts_stage",
        "nts_ticket",
        "lcfs_household",
        "lcfs_person",
        "etb",
        "spi",
        "hmrc_income",
    }
    assert graph.node("lcfs_consumption").sources == (
        "lcfs_household",
        "lcfs_person",
    )
    assert graph.node("hmrc_spi_income_spine").sources == (
        "spi",
        "hmrc_income",
    )
    assert graph.node("hmrc_cgt_gains_spine").sources == ()
    create = graph.node("create_uk_frs")
    assert create.params["sample_fraction"] == 0.1
    assert create.params["sample_seed"] == 999
    assert len(str(create.params["stage_contract_sha256"])) == 64
    assert all(
        len(str(node.params["stage_contract_sha256"])) == 64
        for node in graph.nodes
        if "stage" in node.params
    )


def test_uk_registry_covers_every_kernel_ref_and_hashes_stage_modules() -> None:
    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)

    assert set(registry.refs()) == {node.kernel for node in graph.nodes}
    assert registry.implementation_hash(
        "uk.stage.frs_employment@1"
    ) != registry.implementation_hash("uk.stage.frs_council_tax@1")


def test_uc_relationship_helper_changes_affected_kernel_hashes(monkeypatch) -> None:
    from microcosm.build.uk_runtime import uc_relationships

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    affected_refs = (
        graph.node("create_uk_frs").kernel,
        graph.node("uc_reporter_redraw").kernel,
        graph.node("uc_capital_coherence").kernel,
    )
    control_ref = graph.node("frs_council_tax").kernel
    before = {
        ref: registry.implementation_hash(ref) for ref in (*affected_refs, control_ref)
    }
    helper_path = Path(uc_relationships.__file__).resolve()
    original_read_bytes = Path.read_bytes

    def changed_helper_bytes(path: Path) -> bytes:
        content = original_read_bytes(path)
        if path.resolve() == helper_path:
            return content + b"\n# A helper-only source change.\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_helper_bytes)

    for ref in affected_refs:
        assert registry.implementation_hash(ref) != before[ref]
    assert registry.implementation_hash(control_ref) == before[control_ref]


def test_uk_adapter_source_changes_invalidate_all_consuming_stages(monkeypatch):
    from microcosm.frame.adapters import policyengine_uk

    graph = uk_spine_graph()
    registry = uk_registry(graph=graph)
    stages = (
        "frs_legacy_proxies",
        "frs_education_grant_split",
        "frs_brma",
        "was_wealth",
        "nts_bus_travel",
        "lcfs_consumption",
        "etb_vat",
        "etb_services",
        "uc_reporter_redraw",
    )
    affected = [graph.node(stage).kernel for stage in stages]
    controls = [
        graph.node(stage).kernel
        for stage in ("frs_council_tax", "uc_capital_coherence")
    ]
    before = {ref: registry.implementation_hash(ref) for ref in affected + controls}
    adapter_path = Path(policyengine_uk.__file__).resolve()
    original = Path.read_bytes

    def changed_adapter_bytes(path):
        content = original(path)
        return (
            content + b"\n# adapter-only change\n"
            if path.resolve() == adapter_path
            else content
        )

    monkeypatch.setattr(Path, "read_bytes", changed_adapter_bytes)
    for ref in affected:
        assert registry.implementation_hash(ref) != before[ref]
    for ref in controls:
        assert registry.implementation_hash(ref) == before[ref]


def test_uk_graph_json_round_trip_is_canonical() -> None:
    graph = uk_spine_graph()
    serialized = graph_to_json(graph)

    assert graph_from_json(serialized) == graph
    assert graph_to_json(graph_from_json(serialized)) == serialized


def test_conserve_rejects_a_mass_shift_across_household_sizes() -> None:
    # The executor's ledger is person mass: an expansion that conserves the
    # weight entity's mass but shifts it between households of different size
    # cannot pass ``conserve``.  This is the class that refused the SPI support
    # channel on the licensed FRS 2024-25 spine (68.25m -> 65.44m persons).
    with pytest.raises(PopulationError, match="changed stratum"):
        patch(
            _mixed_size_population(),
            _expand_node(),
            _mass_shifting_expand_result(declared=False),
        )


def test_declared_accepts_the_same_expansion_with_the_kernel_ledger() -> None:
    node = Node(
        id="stack",
        kernel="uk.stage.expand.test@1",
        structural=StructuralDelta.EXPAND,
        base="root",
        params=_expand_node().params,
        mass="declared",
    )
    expanded = patch(
        _mixed_size_population(),
        node,
        _mass_shifting_expand_result(declared=True),
    )

    assert expanded.frame.table("person")["person_household_id"].tolist() == [
        10,
        20,
        20,
        30,
    ]
    assert expanded.frame.weights_for("household").total == pytest.approx(3.0)
    record = expanded.mass_ledger[-1]
    assert record.policy == "declared"
    assert record.before_total == pytest.approx(5.0)
    assert record.after_total == pytest.approx(4.5)


def test_spi_support_channel_declares_its_mass_change_and_cgt_clones_conserve() -> None:
    graph = uk_spine_graph(load_country_spec("uk"))

    assert graph.node("spi_support_channel").mass == "declared"
    assert graph.node("cgt_incidence_clone").mass == "conserve"
    assert graph.node("cgt_band_donors").mass == "free"
    assert graph.node("cgt_incidence_anchor").mass == "conserve"
    assert graph.node("cgt_incidence_anchor").params["expand_cells"] == ()
    assert "cgt_incidence_anchor.owned" not in {node.id for node in graph.nodes}


def test_zero_row_expand_conserves_mass_through_the_executor() -> None:
    """An EXPAND may add no rows and only move weight between incumbent rows."""

    expanded = patch(
        _expand_population(),
        _zero_row_expand_node(),
        _zero_row_expand_result([0.25, 2.75], total=3.0),
    )

    assert expanded.version == "anchor"
    assert expanded.frame.table("person")["person_id"].tolist() == [1, 2]
    assert expanded.frame.table("household")["household_id"].tolist() == [10, 20]
    weights = expanded.frame.weights_for("household")
    assert weights.kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(weights.values, np.array([0.25, 2.75]))
    assert weights.total == 3.0
    record = expanded.mass_ledger[-1]
    assert record.operation == "expand"
    assert record.before_total == 3.0
    assert record.after_total == 3.0
    np.testing.assert_array_equal(
        expanded.design_weights["household"], np.array([1.0, 2.0])
    )
    assert expanded.frame.mass_log[-1].declared_factor == 1.0


def test_zero_row_expand_still_refuses_a_shift_across_household_sizes() -> None:
    # Conservation stays on person mass: a zero-row EXPAND that moves weight
    # between households of different size changes the stratum ledger and is
    # rejected, so the anchor can only move mass within same-size pairs.
    with pytest.raises(PopulationError, match="changed stratum"):
        patch(
            _mixed_size_population(),
            _zero_row_expand_node(),
            _zero_row_expand_result([0.5, 2.5], total=3.0),
        )
