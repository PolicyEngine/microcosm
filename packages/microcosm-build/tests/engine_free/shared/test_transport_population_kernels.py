"""Synthetic H5 graphs exercise the population transport contracts.

Hypothesis invariants: exact CREATE declarations, destination design mass,
boundary mass conservation, deterministic keys/artifacts, zero memoized
executions, inert prose, source content identity, stable seeds and rewrite
identity. The wrapped-function differentials use the same authenticated
synthetic donor, never destination microdata.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.population_kernels import (
    TRANSPORT_BOUNDARY,
    TRANSPORT_CREATE,
    TRANSPORT_CURRENCY,
    TRANSPORT_QUANTILE_MAP,
    register_population_kernels,
)
from microcosm.build.transport.registry import register_transport_population_kernels
from microcosm.frame import WeightKind
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.transport import currency_bridge, quantile_map
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    Node,
    NodeRejectedError,
    Owned,
    SeedSource,
    Slice,
    compile_graph,
)
from microcosm.graph.population import dtype_for_token
from test_support.microcosm_build.transport_graph import (
    canonical_text,
    descendants,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.microcosm_build.transport_population import (
    BAND_FACTS,
    ENTITIES,
    PopulationSources,
    band_document,
    create_node,
    digest,
    direct_population,
    expected_bands,
    population_graph,
    population_registry,
    run_population,
    write_population_sources,
)

PROPERTY = settings(
    max_examples=8,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _keys(run):
    return {name: receipt.key for name, receipt in run.manifest.nodes.items()}


def _artifact_bytes(run):
    return {
        node: {
            name: run.store.load_bytes(key)
            for name, key in receipt.opaque_artifacts.items()
        }
        for node, receipt in run.manifest.nodes.items()
    }


def _structural_columns(entity):
    if entity == "person":
        return {"person_id", "person_household_id", "person_family_id"}
    return {f"{entity}_id"}


class CountingKernel(KernelBase):
    def __init__(self, kernel, calls):
        self.kernel = kernel
        self.calls = calls
        self.ref = kernel.ref
        self.capabilities = kernel.capabilities

    def implementation_hash(self):
        return self.kernel.implementation_hash()

    def run(self, context):
        self.calls.append(self.ref)
        return self.kernel.run(context)


def test_kernel_registration_is_explicit_idempotent_and_country_neutral():
    registry = KernelRegistry()
    assert register_population_kernels(registry) is registry
    assert register_population_kernels(registry) is registry
    kernels = (
        TRANSPORT_CREATE,
        TRANSPORT_BOUNDARY,
        TRANSPORT_CURRENCY,
        TRANSPORT_QUANTILE_MAP,
    )
    assert set(registry.refs()) == {kernel.ref for kernel in kernels}
    for kernel in kernels:
        assert "nz" not in kernel.ref
        assert kernel.capabilities.determinism is Determinism.DETERMINISTIC
        assert kernel.capabilities.consumes_se is False
        assert kernel.implementation_hash() == kernel.implementation_hash()
        assert kernel.capabilities.seed_source is (
            SeedSource.KEYED if kernel is TRANSPORT_CREATE else SeedSource.NONE
        )


def test_aggregate_population_registration_installs_exactly_ten_kernels():
    registry = KernelRegistry()
    assert register_transport_population_kernels(registry) is registry
    assert register_transport_population_kernels(registry) is registry
    assert set(registry.refs()) == {
        "transport.create@1",
        "transport.boundary@1",
        "transport.currency@1",
        "transport.quantile_map@1",
        "transport.unit_attributes@1",
        "concepts.encode@1",
        "concepts.encode_groups@1",
        "takeup.assign@1",
        "geography.support_from_facts@1",
        "transport.scenario_override@1",
    }


@PROPERTY
@example(mass=1)
@given(mass=st.integers(min_value=1, max_value=100_000))
def test_a1_a2_d2_create_contracts_on_a_synthetic_graph(tmp_path, mass):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(root / "sources", mass=float(mass))
        graph = population_graph(sources)
        calls = []
        counted = KernelRegistry()
        registry = population_registry()
        for ref in registry.refs():
            counted.register(CountingKernel(registry.get(ref), calls))
        first = run_population(root / "one", sources, graph=graph, registry=counted)
        assert len(calls) == len(graph.nodes)
        calls.clear()
        second = run_population(root / "two", sources, graph=graph)
        assert _keys(first) == _keys(second)
        assert _artifact_bytes(first) == _artifact_bytes(second)
        assert first.manifest.key == second.manifest.key
        again = run_population(
            root / "one", sources, graph=graph, registry=counted, store=first.store
        )
        assert calls == []
        assert all(receipt.hit for receipt in again.manifest.nodes.values())
        assert _keys(again) == _keys(first)

        create = first.manifest.population("nz.create")
        boundary = first.store.load_frame(first.manifest.nodes["nz.open"].frame_key)
        assert create.schema == NZ_SCHEMA
        assert create.weights_for("household").kind is WeightKind.DESIGN
        for frame in (create, boundary):
            assert frame.resolve_weights("person").values.sum() == pytest.approx(
                mass, rel=1e-6
            )
        declared = {
            (item.entity, item.column) for item in graph.node("nz.create").outputs
        }
        actual = {
            (entity, column)
            for entity in ENTITIES
            for column in create.table(entity)
            if column not in _structural_columns(entity)
        }
        assert actual == declared
        receipt = first.manifest.nodes["nz.create"].receipt
        expected, dropped = direct_population(sources)
        assert dict(receipt["dropped_columns"]) == dropped
        assert receipt["destination_mass"] == mass
        for entity in ENTITIES:
            pd.testing.assert_frame_equal(create.table(entity), expected.table(entity))
        np.testing.assert_array_equal(
            create.weights_for("household").values,
            expected.weights_for("household").values,
        )
        pd.testing.assert_series_equal(create.strata, expected.strata)
        assert create.strata.tolist() == ["us:asec"] * 3 + ["us:puf_tax_detail"] * 3
        # Strata keep object storage whether or not pyarrow is installed.
        assert create.strata.dtype == object


@PROPERTY
@example(prose="Edited prose and citations.")
@given(prose=st.text(alphabet="abcdefghijklmnopqrstuvwxyz .", max_size=35))
def test_a4_description_and_citation_change_no_key(tmp_path, prose):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(root / "sources")
        first = run_population(root / "one", sources)
        second = run_population(
            root / "two", sources, graph=population_graph(sources, description=prose)
        )
        assert _keys(first) == _keys(second)
        assert _artifact_bytes(first) == _artifact_bytes(second)


def _relocated(sources, root):
    root.mkdir(parents=True)
    paths = {}
    for name, original in sources.mapping().items():
        paths[name] = root / f"renamed-{name}.bin"
        shutil.copyfile(original, paths[name])
    return PopulationSources(**paths)


@PROPERTY
@example(source="donor", whitespace=b" ")
@example(source="facts", whitespace=b" ")
@example(source="band_facts", whitespace=b" ")
@given(
    source=st.sampled_from(["donor", "facts", "band_facts"]),
    whitespace=st.sampled_from([b" ", b"\n", b"\t"]),
)
def test_a6_source_paths_are_inert_and_bytes_rekey_exact_consumers(
    tmp_path, source, whitespace
):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(root / "sources")
        first = run_population(root / "run", sources)
        moved = _relocated(sources, root / "moved")
        renamed = run_population(root / "run", moved, store=first.store)
        assert _keys(renamed) == _keys(first)
        assert _artifact_bytes(renamed) == _artifact_bytes(first)
        assert all(receipt.hit for receipt in renamed.manifest.nodes.values())
        changed_path = moved.mapping()[source]
        changed_path.write_bytes(changed_path.read_bytes() + whitespace)
        # The HDF decoder and facts decoder accept trailing whitespace. For
        # the donor, run_population re-pins CREATE (population_graph reads
        # the new size and SHA-256), so CREATE's key moves through its params
        # as well as its source content; stale pins are refused below.
        changed = run_population(root / "run", moved, store=first.store)
        consumer = "nz.bands" if source == "band_facts" else "nz.create"
        expected = descendants(changed.graph, {consumer})
        rekeyed = {
            node for node in _keys(first) if _keys(first)[node] != _keys(changed)[node]
        }
        assert rekeyed == expected
        assert {
            node for node, receipt in changed.manifest.nodes.items() if not receipt.hit
        } == expected


@pytest.mark.parametrize(
    ("edit", "message"),
    [("append", "Donor size mismatch"), ("same_size", "Donor SHA-256 mismatch")],
)
def test_a6_donor_byte_change_with_stale_pins_is_refused(tmp_path, edit, message):
    sources = write_population_sources(tmp_path / "sources")
    stale = population_graph(sources)
    first = run_population(tmp_path / "run", sources, graph=stale)
    payload = sources.donor.read_bytes()
    if edit == "append":
        changed = payload + b" "
    else:
        changed = payload[:-1] + bytes([payload[-1] ^ 1])
    assert changed != payload
    sources.donor.write_bytes(changed)
    with pytest.raises(NodeRejectedError, match=message):
        run_population(tmp_path / "run", sources, graph=stale, store=first.store)


@PROPERTY
@example(permutation=tuple(range(6)), rate=1.6)
@given(
    permutation=st.permutations(tuple(range(6))),
    rate=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
def test_seeds_and_rewrites_preserve_identity_with_wrapped_function_differentials(
    tmp_path, permutation, rate
):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(root / "sources", permutation=permutation)
        graph = population_graph(sources, rate=rate)
        run = run_population(root / "run", sources, graph=graph)
        created = run.manifest.population("nz.create")
        rewritten = run.manifest.population("nz.open")
        direct, _ = direct_population(sources)
        baseline = write_population_sources(root / "baseline")
        stable, _ = direct_population(baseline)
        seeds = created.table("person").set_index("person_id")["take_up_seed"]
        expected_seeds = stable.table("person").set_index("person_id")["take_up_seed"]
        assert ((seeds >= 0) & (seeds < 1)).all()
        pd.testing.assert_series_equal(seeds.sort_index(), expected_seeds.sort_index())
        for entity in ENTITIES:
            for column in _structural_columns(entity):
                np.testing.assert_array_equal(
                    rewritten.table(entity)[column], created.table(entity)[column]
                )
            np.testing.assert_array_equal(
                rewritten.table(entity)[f"{entity}_id"],
                direct.table(entity)[f"{entity}_id"],
            )
        bridged = currency_bridge(
            {entity: direct.table(entity) for entity in ENTITIES},
            {"person": ["interest_income"], "household": ["rent"]},
            rate,
        )
        for entity, column in (("person", "interest_income"), ("household", "rent")):
            np.testing.assert_array_equal(
                rewritten.table(entity)[column], bridged[entity][column]
            )
        bands = expected_bands(sources)
        expected = quantile_map(
            direct.table("person")["employment_income"],
            direct.resolve_weights("person").values,
            bands,
            interpolation="uniform",
        )
        np.testing.assert_array_equal(
            rewritten.table("person")["employment_income"], expected
        )
        # The unequal facts must matter: equal or reversed shares map
        # differently, so the differential checks the facts' shares.
        for shares in (
            [1 / len(bands)] * len(bands),
            [band["share"] for band in reversed(bands)],
        ):
            assert not np.array_equal(
                expected,
                quantile_map(
                    direct.table("person")["employment_income"],
                    direct.resolve_weights("person").values,
                    [
                        {**band, "share": share}
                        for band, share in zip(bands, shares, strict=True)
                    ],
                    interpolation="uniform",
                ),
            )


def _two_holders_in_one_household(tables):
    """Give the first household's second member assets beside its head's."""
    person = tables["person"]
    second = person.index[person["person_household_id"] == 101][1]
    person.loc[second, "bank_account_assets"] = 40


@PROPERTY
@example(mass=1)
@given(mass=st.integers(min_value=1, max_value=100_000))
def test_household_aggregate_quantile_map_equals_direct_function(tmp_path, mass):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(
            root / "sources",
            mass=float(mass),
            band_entity="household",
            edit_tables=_two_holders_in_one_household,
        )
        graph = population_graph(
            sources, qmap_column="liquid_financial_assets", aggregate_entity="household"
        )
        run = run_population(root / "run", sources, graph=graph)
        direct, _ = direct_population(sources)
        person = direct.table("person")
        values = person["liquid_financial_assets"]
        weights = direct.resolve_weights("person").values
        holders = (values != 0).groupby(person["person_household_id"]).sum()
        assert (holders >= 2).any(), "A household needs two nonzero members."
        expected = quantile_map(
            values,
            weights,
            expected_bands(sources),
            interpolation="uniform",
            group={"entity_ids": person["person_household_id"]},
        )
        # Aggregation must matter: mapping persons one by one differs.
        assert not np.array_equal(
            expected,
            quantile_map(
                values, weights, expected_bands(sources), interpolation="uniform"
            ),
        )
        rewritten = run.manifest.population("nz.open").table("person")
        np.testing.assert_array_equal(rewritten["liquid_financial_assets"], expected)
        np.testing.assert_array_equal(rewritten["person_id"], person["person_id"])


@PROPERTY
@example(multipliers=(1, 2))
@given(multipliers=st.tuples(st.integers(1, 4), st.integers(1, 4)))
def test_component_specific_bands_equal_separate_direct_maps(tmp_path, multipliers):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(root / "sources")
        direct, _ = direct_population(sources)
        person = direct.table("person")
        components = sorted(person["sex"].unique().tolist())
        assert len(components) == len(multipliers)
        fixture_bands = expected_bands(sources)
        document = {
            "components": [
                {
                    "value": component,
                    "bands": [
                        {
                            "lower": band["lower"] * multiplier,
                            "upper": band["upper"] * multiplier,
                            "reference": reference,
                        }
                        for band, (reference, _) in zip(
                            fixture_bands, BAND_FACTS, strict=True
                        )
                    ],
                }
                for component, multiplier in zip(components, multipliers, strict=True)
            ]
        }
        graph = population_graph(sources)
        quantile = replace(
            graph.node("nz.quantile"),
            inputs=(Slice("person", ("employment_income", "sex")),),
            params={
                **graph.node("nz.quantile").params,
                "component_column": "sex",
                "bands": canonical_text(document),
                "bands_sha256": digest(document),
            },
        )
        graph = replace(
            graph,
            nodes=tuple(
                quantile if node.id == quantile.id else node for node in graph.nodes
            ),
        )
        run = run_population(root / "run", sources, graph=graph)
        expected = np.empty(len(person), dtype=np.float64)
        weights = direct.resolve_weights("person").values
        for component, multiplier in zip(components, multipliers, strict=True):
            selected = person["sex"].to_numpy() == component
            expected[selected] = quantile_map(
                person["employment_income"].to_numpy()[selected],
                weights[selected],
                [
                    {
                        "lower": band["lower"] * multiplier,
                        "upper": band["upper"] * multiplier,
                        "share": band["share"],
                    }
                    for band in fixture_bands
                ],
                interpolation="uniform",
            )
        rewritten = run.manifest.population("nz.open").table("person")
        np.testing.assert_array_equal(rewritten["employment_income"], expected)
        np.testing.assert_array_equal(rewritten["person_id"], person["person_id"])


@PROPERTY
@example(resource=("nz.currency", "currency_sha256"), digit="0")
@given(
    resource=st.sampled_from(
        [
            ("nz.create", "unit_rule_sha256"),
            ("nz.currency", "currency_sha256"),
            ("nz.quantile", "bands_sha256"),
        ]
    ),
    digit=st.sampled_from("0123456789abcdef"),
)
def test_resource_identity_rekeys_only_its_node_and_consumers(
    tmp_path, resource, digit
):
    """These digests are format-checked only: any hex moves keys, not bytes."""
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(root / "sources")
        before = run_population(root / "run", sources)
        node_id, parameter = resource
        graph = before.graph
        node = replace(
            graph.node(node_id),
            params={**graph.node(node_id).params, parameter: digit * 64},
        )
        edited = replace(
            graph,
            nodes=tuple(node if item.id == node_id else item for item in graph.nodes),
        )
        after = run_population(root / "run", sources, graph=edited, store=before.store)
        assert {
            name for name in _keys(before) if _keys(before)[name] != _keys(after)[name]
        } == descendants(edited, {node_id})
        assert _artifact_bytes(before) == _artifact_bytes(after)


@pytest.mark.parametrize(
    "problem", ["absent_reference", "missing_component", "extra_component"]
)
def test_quantile_map_refuses_absent_facts_and_mismatched_component_coverage(
    tmp_path, problem
):
    sources = write_population_sources(tmp_path / "sources")
    graph = population_graph(sources)
    quantile = graph.node("nz.quantile")
    document = json.loads(quantile.params["bands"])
    if problem == "absent_reference":
        document["bands"][0]["reference"] = "absent_fact"
        inputs = quantile.inputs
        extra_params = {}
        message = "reference is absent"
    else:
        direct, _ = direct_population(sources)
        components = sorted(direct.table("person")["sex"].unique().tolist())
        codes = (
            components[:-1]
            if problem == "missing_component"
            else [*components, "absent_code"]
        )
        document = {
            "components": [
                {"value": code, "bands": document["bands"]} for code in codes
            ]
        }
        inputs = (Slice("person", ("employment_income", "sex")),)
        extra_params = {"component_column": "sex"}
        message = "exactly cover"
    quantile = replace(
        quantile,
        inputs=inputs,
        params={
            **quantile.params,
            "bands": canonical_text(document),
            "bands_sha256": digest(document),
            **extra_params,
        },
    )
    graph = replace(
        graph,
        nodes=tuple(
            quantile if node.id == quantile.id else node for node in graph.nodes
        ),
    )
    with pytest.raises(NodeRejectedError, match=message):
        run_population(tmp_path / "run", sources, graph=graph)


def test_create_runtime_rejects_a_missing_declared_column(tmp_path):
    sources = write_population_sources(tmp_path / "sources")
    graph = population_graph(sources, through="boundary")
    create = replace(
        graph.node("nz.create"), outputs=graph.node("nz.create").outputs[:-1]
    )
    graph = replace(graph, nodes=(create, graph.node("nz.open")))
    compiled = compile_graph(graph)
    assert compiled is not None
    with pytest.raises(NodeRejectedError, match="exactly equal its declaration"):
        run_population(tmp_path / "run", sources, graph=graph)


class StructuralMutation(KernelBase):
    ref = "test.structural_mutation@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        raise AssertionError(
            "The runtime must refuse structural ownership before execution."
        )


@PROPERTY
@example(column="person_household_id")
@given(column=st.sampled_from(["person_id", "person_household_id", "person_family_id"]))
def test_variant_structural_ownership_is_rejected_at_run_time(tmp_path, column):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        root = Path(temporary)
        sources = write_population_sources(root / "sources")
        graph = population_graph(sources, through="boundary")
        node = Node(
            "nz.variant.structural",
            StructuralMutation.ref,
            population="nz.open",
            inputs=(Slice("person", ("age",)),),
            outputs=(Owned("person", column, "int64"),),
        )
        graph = replace(graph, nodes=(*graph.nodes, node))
        compile_graph(graph)
        registry = population_registry()
        registry.register(StructuralMutation())
        with pytest.raises(NodeRejectedError, match="cannot own structural column"):
            run_population(root / "run", sources, graph=graph, registry=registry)


@pytest.mark.parametrize(
    ("entity", "measure", "filter_"),
    [
        ("household", "person_count", None),
        ("person", "employment_income", None),
        ("person", "person_count", "adults"),
    ],
)
def test_create_refuses_a_destination_mass_that_is_not_total_person_population(
    tmp_path, entity, measure, filter_
):
    sources = write_population_sources(tmp_path / "sources")
    document = reference_document(
        [
            reference_row(
                "toy_population", entity=entity, measure=measure, filter_=filter_
            )
        ]
    )
    fact = toy_fact("toy_population", 120.0, entity=entity)
    if measure == "employment_income":
        fact["observed_measure"]["unit"] = "currency"
    write_facts(sources.facts, [fact])
    node = create_node(sources)
    node = replace(
        node,
        params={
            **node.params,
            "mass_reference": canonical_text(document),
            "mass_reference_sha256": digest(document),
        },
    )
    context = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object),
        params=node.params,
        rng=np.random.default_rng(0),
        sources={"donor": sources.donor, "facts": sources.facts},
    )
    with pytest.raises(ValueError, match="population|person count"):
        TRANSPORT_CREATE.run(context)


def test_create_accepts_a_person_count_fact_with_a_country_prepared_measure_name(
    tmp_path,
):
    sources = write_population_sources(tmp_path / "sources")
    document = reference_document(
        [reference_row("toy_population", entity="person", measure="people")]
    )
    node = create_node(sources)
    node = replace(
        node,
        params={
            **node.params,
            "mass_reference": canonical_text(document),
            "mass_reference_sha256": digest(document),
        },
    )
    context = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object),
        params=node.params,
        rng=np.random.default_rng(0),
        sources={"donor": sources.donor, "facts": sources.facts},
    )
    result = TRANSPORT_CREATE.run(context)
    assert result.frame.resolve_weights("person").values.sum() == pytest.approx(120.0)
    assert result.receipt["mass_reference"] == "toy_population"


def _with_band_references(graph, references):
    bands = graph.node("nz.bands")
    bands = replace(
        bands,
        params={
            **bands.params,
            "references": canonical_text(references),
            "references_sha256": digest(references),
        },
    )
    return replace(
        graph,
        nodes=tuple(bands if node.id == bands.id else node for node in graph.nodes),
    )


@pytest.mark.parametrize(
    ("aggregate_entity", "fact_entity", "filter_", "unit"),
    [
        (None, "household", None, "count"),
        ("household", "person", None, "count"),
        (None, "person", "is_female", "count"),
        (None, "person", None, "currency"),
    ],
)
def test_quantile_bands_refuse_facts_that_do_not_count_the_ranked_unit(
    tmp_path, aggregate_entity, fact_entity, filter_, unit
):
    sources = write_population_sources(tmp_path / "sources", band_entity=fact_entity)
    facts = [toy_fact(name, value, entity=fact_entity) for name, value in BAND_FACTS]
    for fact in facts:
        fact["observed_measure"]["unit"] = unit
    write_facts(sources.band_facts, facts)
    graph = population_graph(
        sources,
        qmap_column="employment_income"
        if aggregate_entity is None
        else "liquid_financial_assets",
        aggregate_entity=aggregate_entity,
    )
    references = band_document(fact_entity)
    if filter_ is not None:
        for row in references["target_references"]:
            row["filter"] = filter_
    graph = _with_band_references(graph, references)
    ranked = "person" if aggregate_entity is None else aggregate_entity
    with pytest.raises(NodeRejectedError, match=f"unfiltered {ranked} count facts"):
        run_population(tmp_path / "run", sources, graph=graph)


@pytest.mark.parametrize("problem", ["household_slice", "free_mass"])
def test_boundary_refuses_without_a_person_slice_or_conserved_mass(tmp_path, problem):
    sources = write_population_sources(tmp_path / "sources")
    graph = population_graph(sources, through="boundary")
    boundary = graph.node("nz.open")
    if problem == "household_slice":
        boundary = replace(boundary, inputs=(Slice("household", ("rent",)),))
        message = "person data-column slice"
    else:
        boundary = replace(boundary, mass="free")
        message = "mass='conserve'"
    graph = replace(graph, nodes=(graph.node("nz.create"), boundary))
    with pytest.raises(NodeRejectedError, match=message):
        run_population(tmp_path / "run", sources, graph=graph)


def test_create_artifacts_pin_python_string_storage_with_pyarrow_installed(tmp_path):
    pinned = dtype_for_token("string")
    # microcosm-build depends on pyarrow; with it installed, a bare "string"
    # dtype stores pyarrow strings, which the graph's token does not.
    assert pd.Series(["x"]).astype("string").dtype != pinned
    sources = write_population_sources(tmp_path / "sources")
    run = run_population(tmp_path / "run", sources)
    text = [
        (item.entity, item.column)
        for item in run.graph.node("nz.create").outputs
        if item.dtype == "string"
    ]
    assert ("household", "donor_support_stratum") in text
    for node in ("nz.create", "nz.open"):
        frame = run.manifest.population(node)
        for entity, column in text:
            assert frame.table(entity)[column].dtype == pinned, (node, column)
