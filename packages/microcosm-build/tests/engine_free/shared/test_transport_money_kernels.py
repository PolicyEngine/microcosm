"""``money.normalize@1``, its opt-in producer preflight and the toy lineage run.

Invariants executed here:

- Each declared money column is normalized exactly once per lineage: the
  kernel rewrites it once with its integer companion, composition refuses a
  second normalization in any version's lineage (a FILTER carries every
  column of its base version), and a consumed column needs exactly one
  declared producer.
- Factor authenticity: a factor whose receipt is self-consistent but whose
  operands differ from the authenticated feed refuses on replay.
- The donor money period recorded by CREATE must equal the declaration's.
- A scenario's profile knob selects its own factor (V1 low, V2 high).
- Opt-in: without a money-normalization resource, preflight and composition
  are byte-identical and the toy keeps ``transport.currency@1``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from microcosm.build.reference_factors import (
    resolve_reference_factor,
    validate_factor_context,
)
from microcosm.build.transport import compose as compose_module
from microcosm.build.transport import money_kernels as money_kernels_module
from microcosm.build.transport.artifact_types import (
    INCOME_CELLS_TYPE,
    TARGET_SURFACE_TYPE,
)
from microcosm.build.transport.codecs import register_transport_codecs
from microcosm.build.transport.compose import (
    SKELETON_KERNELS,
    validate_transport_activation,
)
from microcosm.build.transport.money_kernels import MONEY_NORMALIZE
from microcosm.build.transport.registry import register_transport_population_kernels
from microcosm.graph import (
    SOURCE_CODECS,
    ArtifactInput,
    ArtifactOutput,
    ContentStore,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    Slice,
    compile_graph,
    graph_to_json,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.money_normalization import (
    DIMENSION,
    GRAPH_NORMALIZED,
    RAW_FARM,
    RAW_RENTAL,
    ToyMoneyConsumer,
    add_money_columns,
    factor_id,
    graph_declaration,
    money_factor_context,
    money_graph,
    money_registry,
    money_sources,
    normalize_node,
    oracle_factor,
    rate_feed_form,
    write_money_feed,
)
from test_support.microcosm_build.transport_composed import make_composed_fixture
from test_support.microcosm_build.transport_population import (
    write_population_sources,
)

_VALUES = {
    "interest_income": [3.0, 5.0, 0.1],
    "rental_income": [-120.5, 0.0, 35.25],
    "farm_self_employment_income": [0.0, -2.675, 1000000.0],
}


@pytest.fixture(scope="module")
def feed(tmp_path_factory):
    path, pin, identity, facts = write_money_feed(
        tmp_path_factory.mktemp("money-feed"), rate="1.25"
    )
    return path, pin, identity, facts, money_factor_context(identity, facts)


def _oracle_cents(value: float, rate: Fraction, factor: Fraction) -> int:
    return round(round(Fraction(repr(value)) * rate * 100) * factor)


def _context(
    feed,
    *,
    declaration=None,
    factor_context=None,
    selection=None,
    metadata=None,
    pin=True,
    node=None,
):
    path, feed_pin, _, _, context = feed
    declaration = graph_declaration() if declaration is None else declaration
    factor_context = context if factor_context is None else factor_context
    node = node or normalize_node(
        "toy.normalize",
        "toy.central",
        declaration,
        factor_context,
        feed_pin,
        selection=selection,
    )
    params = dict(node.params)
    if not pin:
        params.pop("chronicle_feed_pin")
    return KernelContext(
        node=node,
        tables={
            "person": pd.DataFrame({"person_id": [11, 12, 13], **_VALUES}),
        },
        weights={},
        strata=pd.Series([], dtype=object),
        params=params,
        rng=np.random.default_rng(0),
        sources={"money_facts": path},
        frame_metadata=({"donor_time_period": 2024} if metadata is None else metadata),
    )


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def test_kernel_rewrites_each_column_once_with_its_companion_and_receipt(feed):
    result = MONEY_NORMALIZE.run(_context(feed))
    factor = oracle_factor("central")
    for column, values in _VALUES.items():
        cents = [_oracle_cents(value, Fraction(5, 4), factor) for value in values]
        assert result.columns["person", f"{column}_cents"].tolist() == cents
        assert result.columns["person", column].tolist() == [c / 100 for c in cents]
        assert result.columns["person", column].index.tolist() == [11, 12, 13]
    # 3 * 1.25 * 1.1 = 4.125 ties to even; -2.675 * 1.25 = -3.34375.
    assert result.columns["person", "interest_income_cents"].tolist()[0] == 412
    assert result.columns["person", "farm_self_employment_income_cents"].tolist()[
        1
    ] == round(-334 * Fraction(11, 10))
    receipt = result.receipt
    assert receipt["profile"] == "central"
    assert receipt["recorded_donor_time_period"] == 2024
    assert receipt["rate"]["text"] == "1.25"
    assert {row["factor_id"] for row in receipt["factors"].values()} == {
        factor_id("central")
    }
    assert [row["column"] for row in receipt["columns"]] == GRAPH_NORMALIZED["person"]
    assert all(row["n"] == 3 for row in receipt["columns"])


@pytest.mark.parametrize(
    ("selection", "profile"),
    [
        ({}, "central"),
        ({DIMENSION: "low"}, "low"),
        ({DIMENSION: "high"}, "high"),
        ({DIMENSION: "central", "unrelated": "x"}, "central"),
    ],
)
def test_the_profile_knob_selects_each_variants_own_factor(feed, selection, profile):
    result = MONEY_NORMALIZE.run(_context(feed, selection=selection))
    assert result.receipt["profile"] == profile
    expected = [
        _oracle_cents(value, Fraction(5, 4), oracle_factor(profile))
        for value in _VALUES["rental_income"]
    ]
    assert result.columns["person", "rental_income_cents"].tolist() == expected


def test_an_unknown_profile_refuses(feed):
    with pytest.raises(ValueError, match="Unknown money normalization profile"):
        MONEY_NORMALIZE.run(_context(feed, selection={DIMENSION: "bogus"}))


def test_a_missing_declared_profile_factor_refuses(feed):
    declaration = graph_declaration()
    del declaration["index_family_factors"]["capital_income"]["factors"]["low"]
    with pytest.raises(ValueError, match="one factor for every declared profile"):
        MONEY_NORMALIZE.run(
            _context(feed, declaration=declaration, selection={DIMENSION: "low"})
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"donor_time_period": 2023}, "differs from the declared calendar_year 2024"),
        ({"donor_time_period": "2024"}, "differs from the declared"),
        ({"donor_time_period": None}, "records no money period"),
        ({}, "needs the frame's recorded donor_time_period"),
    ],
)
def test_a_donor_period_mismatch_or_null_refuses(feed, metadata, message):
    with pytest.raises(ValueError, match=message):
        MONEY_NORMALIZE.run(_context(feed, metadata=metadata))


def test_a_self_consistent_tampered_factor_refuses_on_feed_replay(feed):
    _, _, identity, facts, context = feed
    forged_facts = copy.deepcopy(list(facts))
    anchor = next(
        row for row in forged_facts if row["aggregate_fact_key"] == "anchor-0"
    )
    anchor["value"] = 120
    forged = copy.deepcopy(context)
    entry = forged["factors"][1]
    entry["declaration"]["expected_operational_factor"]["numerator"] = 1200000
    factor, receipt = resolve_reference_factor(
        entry["declaration"], forged_facts, identity
    )
    entry["operational_factor"], entry["receipt"] = factor, receipt
    # The forged context is internally consistent ...
    assert set(validate_factor_context(forged)) == {
        factor_id(profile) for profile in ("low", "central", "high")
    }
    # ... but replay against the authenticated feed refuses it.
    with pytest.raises(ValueError, match="expected factor mismatch|replay differs"):
        MONEY_NORMALIZE.run(_context(feed, factor_context=forged))


def test_without_a_feed_pin_the_factor_source_commit_must_be_authenticated(feed):
    with pytest.raises(ValueError, match="cannot be authenticated without a feed pin"):
        MONEY_NORMALIZE.run(_context(feed, pin=False))


def test_a_profile_input_factor_is_never_applied_to_donor_money(feed):
    _, _, identity, facts, _ = feed
    context = money_factor_context(identity, facts, purpose="profile_input")
    with pytest.raises(ValueError, match="applies only money_normalization_input"):
        MONEY_NORMALIZE.run(_context(feed, factor_context=context))


def test_a_feed_selected_rate_replays_through_the_kernel(feed):
    _, _, identity, _, _ = feed
    declaration = graph_declaration()
    declaration["rate"] = rate_feed_form(identity, "1.25")
    result = MONEY_NORMALIZE.run(_context(feed, declaration=declaration))
    assert result.receipt["rate"]["form"] == "feed_selector"
    assert result.receipt["rate"]["operand"]["decimal_operand"] == "1.25"
    assert result.columns["person", "interest_income_cents"].tolist()[0] == 412
    declaration["rate"]["expected_decimal"] = "1.26"
    with pytest.raises(ValueError, match="replay differs"):
        MONEY_NORMALIZE.run(_context(feed, declaration=declaration))


def test_outputs_and_slices_must_equal_the_declared_columns(feed):
    _, pin, _, _, context = feed
    node = normalize_node(
        "toy.normalize", "toy.central", graph_declaration(), context, pin
    )
    short = Node(
        node.id,
        node.kernel,
        population=node.population,
        sources=node.sources,
        inputs=node.inputs,
        outputs=node.outputs[:-1],
        params=node.params,
    )
    with pytest.raises(ValueError, match="rewrite exactly the declared money columns"):
        MONEY_NORMALIZE.run(_context(feed, node=short))
    wide = Node(
        node.id,
        node.kernel,
        population=node.population,
        sources=node.sources,
        inputs=(Slice("person", (*GRAPH_NORMALIZED["person"], "age")),),
        outputs=node.outputs,
        params=node.params,
    )
    with pytest.raises(ValueError, match="slices exactly the declared money columns"):
        MONEY_NORMALIZE.run(_context(feed, node=wide))
    typed = Node(
        node.id,
        node.kernel,
        population=node.population,
        sources=node.sources,
        inputs=node.inputs,
        outputs=node.outputs,
        params=node.params,
        artifact_outputs=(ArtifactOutput("surface", TARGET_SURFACE_TYPE),),
    )
    with pytest.raises(ValueError, match="no artifacts"):
        MONEY_NORMALIZE.run(_context(feed, node=typed))


def test_implementation_hash_binds_the_pure_module_and_factor_replay(monkeypatch):
    seen = []

    def recording(*objects, dependencies=()):
        seen.extend(objects)
        return "0" * 64

    monkeypatch.setattr(money_kernels_module, "source_hash", recording)
    MONEY_NORMALIZE.implementation_hash()
    names = {getattr(item, "__name__", "") for item in seen}
    assert {
        "microcosm.build.money_normalization",
        "microcosm.build.reference_factors",
        "microcosm.build.ledger_artifact",
        "microcosm.build.ledger_targets",
        "microcosm.frame.concepts",
    } <= names
    monkeypatch.undo()
    assert (
        MONEY_NORMALIZE.implementation_hash() == MONEY_NORMALIZE.implementation_hash()
    )


def test_the_kernel_is_registered_and_a_skeleton_kernel():
    registry = KernelRegistry()
    register_transport_population_kernels(registry)
    assert "money.normalize@1" in registry.refs()
    assert "money.normalize@1" in SKELETON_KERNELS
    assert "nz" not in MONEY_NORMALIZE.ref


def test_composition_binds_the_country_feed_pin_to_the_normalizer():
    spec = {
        "country": "xx",
        "resources": {"chronicle_feed": {"source_commit": "c" * 40, "x": 1}},
    }
    params = compose_module._target_feed_params(spec, "money.normalize@1", {})
    assert json.loads(params["chronicle_feed_pin"])["source_commit"] == "c" * 40
    assert "chronicle_feed_pin" not in compose_module._target_feed_params(
        spec, "transport.currency@1", {}
    )


# ---------------------------------------------------------------------------
# The opt-in producer preflight
# ---------------------------------------------------------------------------


def _row(node_id, kernel, population, inputs=(), outputs=(), **fields):
    return {
        "id": node_id,
        "kernel": kernel,
        "population": population,
        "inputs": [{"entity": e, "columns": list(c)} for e, c in inputs],
        "outputs": [
            {"entity": e, "column": c, "dtype": "float64", "rewrite": True}
            for e, c in outputs
        ],
        **fields,
    }


def _structural(node_id, base, kind="filter"):
    return {
        "id": node_id,
        "kernel": "transport.boundary@1",
        "structural": kind,
        "base": base,
    }


_CREATE = {"id": "c", "kernel": "transport.create@1", "structural": "create"}


def _spec(*rows, declaration=None):
    return {
        "country": "xx",
        "resources": {
            "money_normalization": graph_declaration()
            if declaration is None
            else declaration,
            "transport_graph": {"nodes": [_CREATE, _structural("open", "c"), *rows]},
        },
    }


def _normalizer(node_id, population):
    return _row(
        node_id,
        "money.normalize@1",
        population,
        inputs=[("person", GRAPH_NORMALIZED["person"])],
        outputs=[("person", column) for column in GRAPH_NORMALIZED["person"]],
    )


def _reader(node_id, population, *columns, entity="person"):
    return _row(node_id, "concepts.encode@1", population, inputs=[(entity, columns)])


def test_a_normalized_column_read_after_its_normalizer_has_no_gap():
    spec = _spec(
        _structural("central", "open"),
        _normalizer("central.n", "central"),
        _reader("central.read", "central", "interest_income", "rental_income"),
    )
    assert compose_module._money_producer_gaps(spec) == []


def test_a_consumed_pending_column_is_a_named_gap():
    spec = _spec(_reader("open.read", "open", "dividend_income"))
    assert compose_module._money_producer_gaps(spec) == [
        "money column person.dividend_income awaits a donor that carries it "
        "(read by ['open.read'])"
    ]


def test_a_consumed_quantile_mapped_column_without_its_map_names_what_it_awaits():
    spec = _spec(
        _reader("open.read", "open", "employment_income"),
        _reader("open.rent", "open", "rent", entity="household"),
    )
    assert compose_module._money_producer_gaps(spec) == [
        "money column household.rent awaits quantile map <- toy rent bands "
        "(read by ['open.rent'])",
        "money column person.employment_income awaits quantile map <- toy wage "
        "bands (read by ['open.read'])",
    ]


def test_raw_donor_money_read_without_its_normalizer_is_a_gap():
    gaps = compose_module._money_producer_gaps(
        _spec(_reader("open.read", "open", "interest_income"))
    )
    assert gaps == [
        "money column person.interest_income has no money.normalize@1 producer "
        "(read by ['open.read'])"
    ]


def test_a_second_normalization_in_one_lineage_is_a_gap():
    spec = _spec(
        _normalizer("open.n", "open"),
        _structural("central", "open"),
        _normalizer("central.n", "central"),
        _reader("central.read", "central", "interest_income"),
    )
    (gap,) = compose_module._money_producer_gaps(spec)
    assert gap.startswith(
        "money column person.interest_income has 2 producers in one lineage "
        "['central.n', 'open.n']"
    )


def test_variant_lineages_are_independent():
    # A V1-style FILTER on the shared version carries its base, never a sibling.
    spec = _spec(
        _structural("central", "open"),
        _normalizer("central.n", "central"),
        _reader("central.read", "central", "interest_income"),
        _structural("v1", "open"),
        _normalizer("v1.n", "v1"),
        _reader("v1.read", "v1", "interest_income"),
        _structural("v1.after", "v1"),
        _reader("v1.after.read", "v1.after", "interest_income"),
    )
    assert compose_module._money_producer_gaps(spec) == []


def test_only_the_two_money_producers_may_produce_a_money_column():
    spec = _spec(
        _row(
            "open.currency",
            "transport.currency@1",
            "open",
            inputs=[("person", ["interest_income"])],
            outputs=[("person", "interest_income")],
        ),
        _reader("open.read", "open", "interest_income"),
    )
    (gap,) = compose_module._money_producer_gaps(spec)
    assert "produced by transport.currency@1 node 'open.currency'" in gap


def test_a_quantile_map_cannot_produce_a_normalized_column():
    spec = _spec(
        _row(
            "open.qmap",
            "transport.quantile_map@1",
            "open",
            inputs=[("person", ["interest_income"])],
            outputs=[("person", "interest_income")],
        ),
        _reader("open.read", "open", "interest_income"),
    )
    (gap,) = compose_module._money_producer_gaps(spec)
    assert "declared normalized but produced by transport.quantile_map@1" in gap


def test_a_quantile_mapped_drop_default_column_has_its_map_as_sole_producer():
    declaration = graph_declaration()
    declaration["quantile_mapped"]["person"]["public_pension_income"] = {
        "awaits": ["toy pension bands"],
        "reason": "Quantile-mapped although its concept default is drop.",
    }
    spec = _spec(
        _row(
            "open.qmap",
            "transport.quantile_map@1",
            "open",
            inputs=[("person", ["public_pension_income"])],
            outputs=[("person", "public_pension_income")],
        ),
        _reader("open.read", "open", "public_pension_income"),
        declaration=declaration,
    )
    assert compose_module._money_producer_gaps(spec) == []


def test_a_new_destination_money_output_is_derived_instead_of_donor_money():
    derived = _row(
        "open.pension",
        "simulate.rules_by_ref@1",
        "open",
        outputs=[("person", "public_pension_income")],
    )
    derived["outputs"][0]["rewrite"] = False
    spec = _spec(
        derived,
        _reader("open.read", "open", "public_pension_income"),
    )
    assert compose_module._money_producer_gaps(spec) == []


def test_an_invalid_declaration_is_a_named_preflight_gap():
    declaration = graph_declaration()
    declaration["rate"]["text"] = 1.25
    (gap,) = compose_module._money_producer_gaps(_spec(declaration=declaration))
    assert gap.startswith("money_normalization.json: ")


def test_without_the_resource_preflight_reads_nothing():
    spec = {"country": "xx", "resources": {"transport_graph": {"nodes": "malformed"}}}
    assert compose_module._money_producer_gaps(spec) == []
    compose_module._check_single_money_normalization(spec, None, None)


def test_activation_names_money_gaps_only_when_the_toy_opts_in(tmp_path, monkeypatch):
    fixture = make_composed_fixture(tmp_path / "toy")
    validate_transport_activation(fixture.spec)
    graph = fixture.graph()
    kernels = {node.kernel for node in graph.nodes}
    assert "transport.currency@1" in kernels
    assert "money.normalize@1" not in kernels
    plain = graph_to_json(graph)
    extended = graph_to_json(fixture.graph(extended=True))
    # With the hooks removed, composition and activation are byte-identical.
    with monkeypatch.context() as patch:
        patch.setattr(compose_module, "_money_producer_gaps", lambda spec: [])
        patch.setattr(
            compose_module, "_check_single_money_normalization", lambda *args: None
        )
        assert graph_to_json(fixture.graph()) == plain
        assert graph_to_json(fixture.graph(extended=True)) == extended
        validate_transport_activation(fixture.spec)
    opted = copy.deepcopy(fixture.spec)
    opted["resources"]["money_normalization"] = graph_declaration()
    with pytest.raises(ValueError) as error:
        validate_transport_activation(opted)
    assert (
        "money column person.interest_income is produced by transport.currency@1 "
        "node 'nz.transport.currency'"
    ) in str(error.value)


# ---------------------------------------------------------------------------
# The toy lineage run: CREATE -> open -> {central, v1}
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory, feed):
    root = tmp_path_factory.mktemp("money-graph")
    population = write_population_sources(
        root / "population", time_period=2024, edit_tables=add_money_columns
    )
    path, pin, _, _, context = feed
    declaration = graph_declaration()
    register_transport_codecs(SOURCE_CODECS)
    graph = money_graph(population, declaration, context, pin)
    manifest = run_graph(
        compile_graph(graph),
        sources=money_sources(population, path),
        store=ContentStore(root / "store"),
        kernels=money_registry(),
    )
    return population, declaration, context, pin, graph, manifest


def test_each_branch_normalizes_once_with_its_own_factor(toy_run):
    population, declaration, _, _, graph, manifest = toy_run
    spec = {"country": "xx", "resources": {"money_normalization": declaration}}
    compose_module._check_single_money_normalization(spec, graph, compile_graph(graph))
    created = manifest.population("nz.create").table("person")
    raw = {column: created[column].tolist() for column in GRAPH_NORMALIZED["person"]}
    assert raw["rental_income"] == RAW_RENTAL
    assert raw["farm_self_employment_income"] == RAW_FARM
    for branch, profile in (
        ("toy.central", "central"),
        ("toy.v1", "low"),
        ("toy.v2", "high"),
    ):
        receipt = manifest.nodes[f"{branch}.consume"].receipt
        factor = oracle_factor(profile)
        for column in GRAPH_NORMALIZED["person"]:
            cents = [
                _oracle_cents(value, Fraction(5, 4), factor) for value in raw[column]
            ]
            assert list(receipt[f"{column}_cents"]) == cents, (branch, column)
            assert list(receipt[column]) == [c / 100 for c in cents]
        normalize = manifest.nodes[f"{branch}.normalize"].receipt
        assert normalize["profile"] == profile
        assert normalize["recorded_donor_time_period"] == 2024
    # The shared version is untouched: normalization lives on each branch.
    assert (
        manifest.population("toy.open").table("person")["rental_income"].tolist()
        == RAW_RENTAL
    )
    assert "rental_income_cents" not in manifest.population("toy.open").table("person")


def test_create_records_the_donor_money_period_for_the_branches(toy_run):
    *_, manifest = toy_run
    assert manifest.nodes["nz.create"].receipt["donor_time_period"] == 2024
    for version in ("nz.create", "toy.open", "toy.central", "toy.v1", "toy.v2"):
        assert manifest.population(version).metadata["donor_time_period"] == 2024


def test_a_second_normalization_in_a_shared_version_refuses_at_composition(toy_run):
    population, declaration, context, pin, _, _ = toy_run
    doubled = money_graph(population, declaration, context, pin, normalize_open=True)
    spec = {"country": "xx", "resources": {"money_normalization": declaration}}
    with pytest.raises(ValueError, match="normalized more than once in the lineage"):
        compose_module._check_single_money_normalization(
            spec, doubled, compile_graph(doubled)
        )


class _ToyBaselineCells(ToyMoneyConsumer):
    """Read normalized branch money and exchange baseline reporting evidence."""

    ref = "toy.baseline_cells@1"

    def run(self, context):
        result = super().run(context)
        if context.artifacts:
            baseline = json.loads(context.artifacts["baseline_cells"].payload)
            return KernelResult(
                receipt={**result.receipt, "baseline_cells": baseline["person_id"]}
            )
        return replace(
            result,
            artifacts={"baseline_cells": canonical_json(dict(result.receipt))},
        )


def test_s0_v1_normalize_once_despite_a_cross_population_baseline_artifact(
    tmp_path, toy_run, feed
):
    population, declaration, _, _, graph, _ = toy_run
    baseline = replace(
        graph.node("toy.central.consume"),
        kernel=_ToyBaselineCells.ref,
        artifact_outputs=(ArtifactOutput("baseline_cells", INCOME_CELLS_TYPE),),
    )
    variant = replace(
        graph.node("toy.v1.consume"),
        kernel=_ToyBaselineCells.ref,
        artifact_inputs=(
            ArtifactInput(
                "baseline_cells", baseline.id, "baseline_cells", INCOME_CELLS_TYPE
            ),
        ),
    )
    graph = replace(
        graph,
        nodes=tuple(
            baseline
            if node.id == baseline.id
            else variant
            if node.id == variant.id
            else node
            for node in graph.nodes
        ),
    )
    compiled = compile_graph(graph)
    # The artifact makes both normalizers DAG ancestors of V1's consumer.
    # It does not make S0 a structural base of V1 or transfer its ownership.
    ancestors = compose_module._ancestor_closure(compiled.predecessors, {variant.id})
    assert {"toy.central.normalize", "toy.v1.normalize"} <= ancestors
    spec = {"country": "xx", "resources": {"money_normalization": declaration}}
    compose_module._check_single_money_normalization(spec, graph, compiled)
    registry = money_registry()
    registry.register(_ToyBaselineCells())
    manifest = run_graph(
        compiled,
        sources=money_sources(population, feed[0]),
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
    )
    assert (
        manifest.nodes[variant.id].receipt["baseline_cells"]
        == manifest.nodes[baseline.id].receipt["person_id"]
    )
    assert manifest.nodes["toy.v1.normalize"].receipt["profile"] == "low"
    assert manifest.nodes["toy.central.normalize"].receipt["profile"] == "central"


def test_currency_then_normalization_refuses_on_the_same_money_lineage(toy_run):
    population, declaration, context, pin, _, _ = toy_run
    graph = money_graph(population, declaration, context, pin)
    currency = Node(
        "toy.open.currency",
        "transport.currency@1",
        population="toy.open",
        inputs=(Slice("person", ("interest_income",)),),
        outputs=(Owned("person", "interest_income", "float64", rewrite=True),),
    )
    graph = replace(graph, nodes=(*graph.nodes, currency))
    spec = {"country": "xx", "resources": {"money_normalization": declaration}}
    with pytest.raises(ValueError, match="normalized more than once.*interest_income"):
        compose_module._check_single_money_normalization(
            spec, graph, compile_graph(graph)
        )


def test_derived_money_does_not_add_a_donor_conversion_to_a_compiled_lineage(toy_run):
    _, declaration, _, _, graph, _ = toy_run
    derived = Node(
        "toy.central.pension",
        "simulate.rules_by_ref@1",
        population="toy.central",
        inputs=(Slice("person", ("interest_income",)),),
        outputs=(Owned("person", "public_pension_income", "float64"),),
    )
    reader = Node(
        "toy.central.pension.read",
        "toy.money_consumer@1",
        population="toy.central",
        inputs=(Slice("person", ("public_pension_income",)),),
    )
    graph = replace(graph, nodes=(*graph.nodes, derived, reader))
    spec = {"country": "xx", "resources": {"money_normalization": declaration}}
    compose_module._check_single_money_normalization(spec, graph, compile_graph(graph))


def test_each_compiled_producer_must_match_its_declared_route(toy_run):
    population, declaration, context, pin, graph, _ = toy_run
    qmap = Node(
        "toy.v1.qmap",
        "transport.quantile_map@1",
        population="toy.v1",
        inputs=(Slice("person", ("employment_income",)),),
        outputs=(Owned("person", "employment_income", "float64", rewrite=True),),
    )
    spec = {"country": "xx", "resources": {"money_normalization": declaration}}
    graph_with_map = type(graph)(graph.country, graph.sources, (*graph.nodes, qmap))
    compose_module._check_single_money_normalization(
        spec, graph_with_map, compile_graph(graph_with_map)
    )
    wrong = copy.deepcopy(declaration)
    wrong["quantile_mapped"]["person"].pop("employment_income")
    wrong["pending"]["person"]["employment_income"] = {
        "awaits": ["x"],
        "reason": "x",
    }
    with pytest.raises(ValueError, match="declared pending but"):
        compose_module._check_single_money_normalization(
            {"country": "xx", "resources": {"money_normalization": wrong}},
            graph_with_map,
            compile_graph(graph_with_map),
        )


def test_a_toy_graph_without_a_donor_period_refuses_at_the_normalizer(tmp_path, feed):
    population = write_population_sources(
        tmp_path / "population", edit_tables=add_money_columns
    )
    path, pin, _, _, context = feed
    register_transport_codecs(SOURCE_CODECS)
    graph = money_graph(population, graph_declaration(), context, pin)
    with pytest.raises(Exception, match="records no money period"):
        run_graph(
            compile_graph(graph),
            sources=money_sources(population, path),
            store=ContentStore(tmp_path / "store"),
            kernels=money_registry(),
        )
