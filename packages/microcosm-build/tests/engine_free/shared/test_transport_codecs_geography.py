"""Synthetic donor codecs and facts-derived atomic support.

Properties: codec registration is idempotent; tree bytes ignore the root path
and file creation order; donor columns, original design mass, row order and
support channels match the shared reader; atomic counts conserve each published
population, differ from exact fractional quotas by less than one, and ignore
row permutation. Differential tests compare codec/kernel outputs with their
wrapped reader and support builder. No real donor or destination data is used.
"""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.graph_atomic_geography import (
    AtomicAssignKernel,
    AtomicSupportImportKernel,
)
from microcosm.build.ledger_artifact import (
    CONSUMER_ARTIFACT_SCHEMA_VERSION,
    load_ledger_consumer_artifact,
)
from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.codecs import (
    DONOR_SOURCE_CODEC,
    FACTS_SOURCE_CODEC,
    RULESPEC_SOURCE_CODEC,
    load_ledger_consumer_bytes,
    load_populace_us_h5,
    load_rulespec_tree_bytes,
    register_transport_codecs,
)
from microcosm.build.transport.geography_kernels import (
    ATOMIC_SUPPORT_TYPE,
    GEOGRAPHY_SUPPORT_FROM_FACTS,
    register_geography_kernels,
    support_from_surface,
)
from microcosm.build.transport.target_kernels import (
    decode_target_surface,
    encode_target_surface,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.hierarchy import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
)
from microcosm.frame import WeightKind
from microcosm.frame.transport import read_populace_us_donor
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    SourceRef,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry
from microcosm.graph.keys import source_content_key
from test_support.microcosm_build.transport_graph import toy_fact, write_facts
from test_support.paths import paths_for

PROPERTY = settings(max_examples=40, deadline=None, database=None)
FILE_PROPERTY = settings(
    max_examples=8,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
SHA = "4" * 64


def _surface(populations=(100, 61), *, unit="count") -> bytes:
    specs, references = [], []
    for index, population in enumerate(populations):
        code = f"0{index + 1}"
        name = f"population_{code}"
        specs.append(
            TargetSpec(
                name=name,
                entity="person",
                measure="people",
                value=population,
                source="Synthetic population publication",
                family="demography",
                metadata={
                    "precal_use": "atomic_geography_support",
                    "ledger_measure_unit": unit,
                },
                hierarchy=CalibrationHierarchy(
                    provider=HierarchyNode("toy", "Toy"),
                    category=HierarchyCategory("population", "Population", "toy"),
                    geography=HierarchyGeography(
                        code, f"Toy area {code}", "territorial_authority"
                    ),
                    dimensions=(),
                    target=HierarchyNode(name, name),
                ),
            )
        )
        references.append(
            LedgerTargetReference(
                name=name,
                ledger_fact_key=f"toy:{name}",
                entity="person",
                measure="people",
                metadata={"activation_status": "active"},
            )
        )
    return encode_target_surface(
        TargetRegistry(specs, country="xx"),
        references,
        references_sha256=SHA,
        facts={"facts_sha256": SHA},
    )


def _definition(shares=(0.3, 0.7)) -> dict:
    codes = ("area", "ta", "region", "as_area", "as_area_alt")
    columns = {
        column: {
            "kind": "code",
            "source": "synthetic-crosswalk",
            "vintage": "toy-2025",
            "relation": "exact",
        }
        for column in codes
    }
    columns["population"] = {
        "kind": "weight",
        "source": "synthetic-facts",
        "basis": "persons",
    }
    return {
        "version": 1,
        "level": "atomic",
        "code_system": "toy",
        "vintage": "toy-2025",
        "columns": columns,
        "share_tolerance": 1e-9,
        "population_geography_level": "territorial_authority",
        "population_code_column": "ta",
        "rows": [
            {
                "target": f"population_0{index + 1}",
                "codes": {
                    "area": f"0{index + 1}:{area}",
                    "ta": f"0{index + 1}",
                    "region": "R" if index == 0 else "S",
                    "as_area": str(area),
                    "as_area_alt": str(1 + shares.index(max(shares))),
                },
                "population_share": share,
            }
            for index in range(2)
            for area, share in enumerate(shares, 1)
        ],
    }


def _node(definition=None, **params) -> Node:
    return Node(
        "geo.support",
        GEOGRAPHY_SUPPORT_FROM_FACTS.ref,
        population="nz.open",
        artifact_inputs=(
            ArtifactInput("surface", "precal", "surface", TARGET_SURFACE_TYPE),
        ),
        artifact_outputs=(ArtifactOutput("support", ATOMIC_SUPPORT_TYPE),),
        params={
            "definition": canonical_json(
                _definition() if definition is None else definition
            ).decode(),
            "definition_sha256": SHA,
            "system": "toy-atomic",
            **params,
        },
    )


def _context(node=None, payload=None) -> KernelContext:
    node = _node() if node is None else node
    payload = _surface() if payload is None else payload
    return KernelContext(
        node=node,
        params=node.params,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object),
        rng=np.random.default_rng(0),
        sources={},
        artifacts={
            "surface": ArtifactValue(
                payload, TARGET_SURFACE_TYPE, SHA, SHA, NumericScope()
            )
        },
    )


def test_codec_registration_is_explicit_idempotent_and_mode_correct() -> None:
    registry = SourceCodecRegistry()
    assert registry.names() == registry.bytes_names() == ()
    register_transport_codecs(registry)
    before = registry.as_mapping(), registry.as_bytes_mapping()
    register_transport_codecs(registry)
    assert before == (registry.as_mapping(), registry.as_bytes_mapping())
    assert registry.names() == (DONOR_SOURCE_CODEC,)
    assert registry.bytes_names() == tuple(
        sorted((FACTS_SOURCE_CODEC, RULESPEC_SOURCE_CODEC))
    )
    with pytest.raises(TypeError, match="Frame codec"):
        registry.load_bytes(DONOR_SOURCE_CODEC, "unused")
    with pytest.raises(TypeError, match="raw-bytes codec"):
        registry.load(FACTS_SOURCE_CODEC, "unused")
    conflicting = SourceCodecRegistry()
    conflicting.register_bytes(FACTS_SOURCE_CODEC, lambda path, **kwargs: b"wrong")
    with pytest.raises(ValueError, match="already registered"):
        register_transport_codecs(conflicting)


@FILE_PROPERTY
@given(permutation=st.permutations(tuple(range(6))))
def test_donor_codec_matches_reader_columns_weights_channels_and_row_order(
    tmp_path, permutation
) -> None:
    fixture = (
        paths_for("microcosm-frame").tests / "fixtures/transport/synthetic_donor.json"
    )
    raw = {
        entity: pd.DataFrame(rows)
        for entity, rows in json.loads(fixture.read_text())["tables"].items()
    }
    raw["person"] = raw["person"].iloc[list(permutation)].reset_index(drop=True)
    path = tmp_path / "donor.h5"
    with pd.HDFStore(path, mode="w") as store:
        for entity, table in raw.items():
            store.put(entity, table, format="table")
    direct = read_populace_us_donor(
        path, sha256=sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size
    )
    frame = load_populace_us_h5(path)
    for entity, table in direct.tables.items():
        pd.testing.assert_frame_equal(frame.table(entity), table)
    assert frame.weights_for("household").kind == WeightKind.DESIGN
    np.testing.assert_array_equal(frame.weights_for("household").values, direct.weights)
    household_rows = pd.Index(direct.tables["household"]["household_id"]).get_indexer(
        frame.person["person_household_id"]
    )
    assert frame.strata.tolist() == direct.support_strata[household_rows].tolist()
    assert frame.metadata["donor_country"] == "us"
    assert frame.metadata["currency"] == "USD"
    assert frame.metadata["source_person_ids"] == tuple(direct.source_person_ids)


@FILE_PROPERTY
@given(value=st.integers(min_value=0, max_value=1_000_000))
def test_facts_codec_matches_validated_feed_bytes_and_refuses_tampering(
    tmp_path, value
) -> None:
    path = write_facts(
        tmp_path / "facts.jsonl", [toy_fact("population", value, entity="person")]
    )
    payload = path.read_bytes()
    direct = load_ledger_consumer_artifact(path)
    assert direct.facts[0]["value"] == value
    assert load_ledger_consumer_bytes(path) == payload
    directory = tmp_path / "artifact"
    directory.mkdir(exist_ok=True)
    (directory / "consumer_facts.jsonl").write_bytes(payload)
    (directory / "manifest.json").write_bytes(
        canonical_json(
            {
                "schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
                "facts_sha256": sha256(payload).hexdigest(),
                "fact_row_count": 1,
            }
        )
    )
    assert load_ledger_consumer_bytes(directory) == payload
    (directory / "consumer_facts.jsonl").write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="manifest hash"):
        load_ledger_consumer_bytes(directory)


@FILE_PROPERTY
@given(first=st.binary(max_size=64), second=st.binary(max_size=64))
def test_rulespec_codec_is_root_path_and_creation_order_inert_and_byte_sensitive(
    tmp_path, first, second
) -> None:
    trees = [tmp_path / "first", tmp_path / "renamed"]
    for tree, files in zip(
        trees, (("b.yaml", "a.yaml"), ("a.yaml", "b.yaml")), strict=True
    ):
        tree.mkdir(exist_ok=True)
        for name in files:
            (tree / name).write_bytes(first if name == "a.yaml" else second)
    payload = load_rulespec_tree_bytes(trees[0])
    assert load_rulespec_tree_bytes(trees[1]) == payload
    assert source_content_key("rules", trees[0]) == source_content_key(
        "rules", trees[1]
    )
    document = json.loads(payload)
    assert [file["path"] for file in document["files"]] == ["a.yaml", "b.yaml"]
    assert base64.b64decode(document["files"][0]["bytes_base64"]) == first
    (trees[1] / "a.yaml").write_bytes(first + b"x")
    assert load_rulespec_tree_bytes(trees[1]) != payload
    assert source_content_key("rules", trees[0]) != source_content_key(
        "rules", trees[1]
    )


@FILE_PROPERTY
@example(order=(3, 2, 1, 0))
@given(
    order=st.permutations(tuple(range(4))).filter(lambda order: order != (0, 1, 2, 3))
)
def test_rulespec_codec_sorts_whatever_order_the_directory_lists(
    tmp_path, order
) -> None:
    """The envelope sorts entries itself; listing order never reaches it."""

    tree = tmp_path / "tree"
    names = ("a.yaml", "b/c.yaml", "b/d.yaml", "e.yaml")
    for name in names:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_bytes(name.encode())
    payload = load_rulespec_tree_bytes(tree)
    listing = Path.rglob

    def reordered(self, pattern, *args, **kwargs):
        entries = sorted(listing(self, pattern, *args, **kwargs))
        files = [entry for entry in entries if entry.is_file()]
        # Directories first, then the files in a non-sorted order.
        return iter(
            [
                *(entry for entry in entries if not entry.is_file()),
                *(files[index] for index in order),
            ]
        )

    with mock.patch.object(Path, "rglob", reordered):
        assert load_rulespec_tree_bytes(tree) == payload
    document = json.loads(payload)
    assert [file["path"] for file in document["files"]] == sorted(names)


def test_rulespec_codec_refuses_empty_tree_file_and_symlink(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        load_rulespec_tree_bytes(tmp_path)
    file = tmp_path / "file"
    file.write_bytes(b"x")
    with pytest.raises(ValueError, match="regular directory"):
        load_rulespec_tree_bytes(file)
    (tmp_path / "link").symlink_to(file)
    with pytest.raises(ValueError, match="symlinks"):
        load_rulespec_tree_bytes(tmp_path)


@PROPERTY
@given(
    populations=st.tuples(st.integers(1, 1_000_000), st.integers(1, 1_000_000)),
    numerator=st.integers(0, 100),
    permutation=st.permutations(tuple(range(4))),
)
def test_support_conserves_facts_apportions_within_one_and_ignores_row_order(
    populations, numerator, permutation
) -> None:
    shares = (numerator / 100, (100 - numerator) / 100)
    definition = _definition(shares)
    surface = decode_target_surface(_surface(populations))
    payload = support_from_surface(surface, definition, system="toy-atomic")
    support = decode_atomic_support(payload)
    for index, population in enumerate(populations):
        selected = support.arrays["ta"] == f"0{index + 1}"
        actual = support.arrays["population"][selected]
        assert sum(int(count) for count in actual) == population
        denominator = sum((Fraction(str(share)) for share in shares), Fraction())
        for count, share in zip(actual, shares, strict=True):
            assert (
                abs(
                    Fraction(int(count))
                    - population * Fraction(str(share)) / denominator
                )
                < 1
            )
    permuted = deepcopy(definition)
    permuted["rows"] = [definition["rows"][index] for index in permutation]
    assert support_from_surface(surface, permuted, system="toy-atomic") == payload
    for array in support.arrays.values():
        assert not array.flags.writeable


def test_geography_kernel_declares_its_siblings_platform_bitwise_numerics() -> None:
    """Its deflated NPZ payload records the creating platform (zip header)."""

    numeric = GEOGRAPHY_SUPPORT_FROM_FACTS.capabilities.numeric
    assert numeric is Numeric.PLATFORM_BITWISE
    assert numeric is AtomicSupportImportKernel.capabilities.numeric
    assert numeric is AtomicAssignKernel.capabilities.numeric


def test_geography_kernel_is_differential_to_shared_support_builder() -> None:
    result = GEOGRAPHY_SUPPORT_FROM_FACTS.run(_context())
    assert result.artifacts["support"] == support_from_surface(
        decode_target_surface(_surface()), _definition(), system="toy-atomic"
    )
    assert result.receipt["definition_sha256"] == SHA
    assert result.receipt["areas"] == 4
    assert GEOGRAPHY_SUPPORT_FROM_FACTS.capabilities.consumes_se is False
    assert (
        GEOGRAPHY_SUPPORT_FROM_FACTS.implementation_hash()
        == GEOGRAPHY_SUPPORT_FROM_FACTS.implementation_hash()
    )


@pytest.mark.parametrize(
    "defect",
    (
        "empty",
        "missing_target",
        "duplicate_area",
        "bad_share",
        "bad_sum",
        "wrong_code",
        "fractional_population",
        "wrong_unit",
        "wrong_metadata",
    ),
)
def test_support_refuses_unharvested_incomplete_and_ambiguous_data(defect) -> None:
    definition = _definition()
    payload = _surface()
    if defect == "empty":
        definition["rows"] = []
    elif defect == "missing_target":
        definition["rows"] = definition["rows"][:2]
    elif defect == "duplicate_area":
        definition["rows"][1]["codes"]["area"] = definition["rows"][0]["codes"]["area"]
    elif defect == "bad_share":
        definition["rows"][0]["population_share"] = -1
    elif defect == "bad_sum":
        definition["rows"][0]["population_share"] = 0.1
    elif defect == "wrong_code":
        definition["rows"][0]["codes"]["ta"] = "unknown"
    elif defect == "fractional_population":
        payload = _surface((100.5, 61))
    elif defect == "wrong_unit":
        payload = _surface(unit="currency")
    else:
        definition["columns"]["as_area"]["relation"] = "guess"
    with pytest.raises(ValueError):
        GEOGRAPHY_SUPPORT_FROM_FACTS.run(_context(_node(definition), payload))


@pytest.mark.parametrize(
    "params",
    ({"definition_sha256": "bad"}, {"unused": "x"}, {"definition": '{"version": 1}'}),
)
def test_geography_kernel_refuses_unknown_unbound_or_noncanonical_params(
    params,
) -> None:
    with pytest.raises(ValueError):
        GEOGRAPHY_SUPPORT_FROM_FACTS.run(_context(_node(**params)))


class _SurfaceSource(KernelBase):
    ref = "test.surface_source@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        return KernelResult(
            artifacts={"surface": context.sources["geo_surface"].read_bytes()}
        )


def test_geography_kernel_runs_after_synthetic_h5_create_and_memoizes(tmp_path) -> None:
    from test_support.microcosm_build.transport_population import (
        population_graph,
        population_registry,
        write_population_sources,
    )

    sources = write_population_sources(tmp_path / "inputs")
    base = population_graph(sources, through="boundary")
    surface_path = tmp_path / "surface.json"
    surface_path.write_bytes(_surface())
    surface_node = Node(
        "precal",
        _SurfaceSource.ref,
        population="nz.open",
        sources=("geo_surface",),
        artifact_outputs=(ArtifactOutput("surface", TARGET_SURFACE_TYPE),),
    )
    graph = compile_graph(
        Graph(
            base.country,
            (*base.sources, SourceRef("geo_surface", "raw-bytes-v1")),
            (*base.nodes, surface_node, _node()),
        )
    )
    registry = population_registry()
    registry.register(_SurfaceSource())
    register_geography_kernels(registry)
    mapping = {**sources.mapping(), "geo_surface": surface_path}
    store = ContentStore(tmp_path / "store")
    cold = run_graph(graph, sources=mapping, store=store, kernels=registry)
    warm = run_graph(
        graph, sources=mapping, store=store, kernels=registry, resume="require"
    )
    output = cold.nodes["geo.support"].opaque_artifacts["support"]
    expected = GEOGRAPHY_SUPPORT_FROM_FACTS.run(_context()).artifacts["support"]
    assert store.load_bytes(output) == expected
    assert all(node.hit for node in warm.nodes.values())
    fresh = ContentStore(tmp_path / "fresh")
    again = run_graph(graph, sources=mapping, store=fresh, kernels=registry)
    assert cold.nodes["geo.support"].key == again.nodes["geo.support"].key
    assert (
        fresh.load_bytes(again.nodes["geo.support"].opaque_artifacts["support"])
        == expected
    )
    inert = compile_graph(
        replace(
            graph.graph,
            nodes=tuple(
                replace(node, description="Synthetic citation-only description")
                for node in graph.graph.nodes
            ),
        )
    )
    edited = run_graph(
        inert, sources=mapping, store=store, kernels=registry, resume="require"
    )
    assert all(node.hit for node in edited.nodes.values())
    renamed = tmp_path / "renamed-surface.json"
    renamed.write_bytes(surface_path.read_bytes())
    renamed_mapping = {**mapping, "geo_surface": renamed}
    moved = run_graph(
        graph, sources=renamed_mapping, store=store, kernels=registry, resume="require"
    )
    assert all(node.hit for node in moved.nodes.values())
    renamed.write_bytes(_surface((101, 61)))
    changed = run_graph(graph, sources=renamed_mapping, store=store, kernels=registry)
    assert {name for name, node in changed.nodes.items() if not node.hit} == {
        "precal",
        "geo.support",
    }
