"""Invented-only raw Schedule C transfer; no genuine source or tax calculation."""

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_combined_clone as clone
from microcosm.build.us_runtime import graph_puf_detail_transfer as graph_detail
from microcosm.build.us_runtime import puf_detail_transfer as detail
from microcosm.build.us_runtime import puf_growth_graph as growth_graph
from microcosm.build.us_runtime import puf_monetary_agi_projection as agi
from microcosm.build.us_runtime import puf_raw_source as raw
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_qrf import LegacyQRFTrainKernel, legacy_qrf_train_nodes
from microcosm.fit.model_input import decode_recipient_matrix
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import SourceCodecRegistry, load_raw_bytes
from microcosm.graph.population import Population, dtype_for_token

_fixture_spec = importlib.util.spec_from_file_location(
    "puf_detail_source_fixture",
    Path(__file__).with_name("test_us_puf_price_baseline.py"),
)
source_fixture = importlib.util.module_from_spec(_fixture_spec)
sys.modules[_fixture_spec.name] = source_fixture
_fixture_spec.loader.exec_module(source_fixture)

HERE = Path(__file__).resolve().parents[3]
BOUNDARY = "fixture_puf_detail.population"


def arm(channel, wage_delta=0):
    if channel == "asec":
        pids, households, tax_units = [1, 2, 3], [11, 11, 42], [7, 7, 15]
        head, spouse, dependent = [True, False, True], [False] * 3, [False, True, False]
        wages, statuses, hh_ids, weights = (
            [10.0 + wage_delta, 999.0, 20.0],
            ["SURVIVING_SPOUSE", "SINGLE"],
            [11, 42],
            [13.0, 0.0],
        )
        ages = [40, 12, 50]
    else:
        pids, households, tax_units = [1, 2, 4, 7], [11] * 4, [7, 7, 7, 8]
        head, spouse, dependent = (
            [True, False, False, True],
            [False, True, False, False],
            [False, False, True, False],
        )
        wages, statuses, hh_ids, weights = (
            [100.0 + wage_delta, 70.0, 800.0, 30.0],
            ["JOINT", "SINGLE"],
            [11],
            [7.0],
        )
        ages = [40, 39, 12, 21]
    unique_units = list(dict.fromkeys(tax_units))
    person = pd.DataFrame(
        {
            "person_id": pids,
            "person_household_id": households,
            "person_tax_unit_id": tax_units,
            "person_spm_unit_id": households,
            "person_family_id": households,
            "person_marital_unit_id": pids,
            "age": ages,
            "is_tax_unit_head": head,
            "is_tax_unit_spouse": spouse,
            "is_tax_unit_dependent": dependent,
            "source_year": pd.Series(
                ["2024"] * len(pids), dtype=dtype_for_token("string")
            ),
            (
                "asec_reported_wage_income_2024_price"
                if channel == "asec"
                else "employment_income_before_lsr"
            ): wages,
        }
    )
    person[channel + "_source_only"] = pd.Series(
        [pd.NA, *range(1, len(pids))], dtype="Int64"
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": hh_ids, "state_fips": [6] * len(hh_ids)}
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": unique_units,
                    "filing_status_input": pd.Series(
                        statuses, dtype=dtype_for_token("string")
                    ),
                }
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": hh_ids}),
            "family": pd.DataFrame({"family_id": hh_ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": pids}),
        },
        US_SCHEMA,
        {"household": Weights(np.array(weights), WeightKind.DESIGN)},
        pd.Series([channel] * len(pids), dtype=dtype_for_token("string")),
    )


def host(wage_delta=0):
    result = assemble_spines(
        {"asec": arm("asec", wage_delta), "acs": arm("acs", wage_delta)},
        household_mass_shares={"asec": 0.6, "acs": 0.4},
    )
    tables = {}
    for e in result.entities:
        table = result.table(e).copy()
        for col in table:
            if not pd.api.types.is_numeric_dtype(table[col]):
                table[col] = table[col].astype(dtype_for_token("string"))
        tables[e] = table
    return Frame(
        tables,
        US_SCHEMA,
        {"household": result.weights_for("household")},
        result.strata.astype(dtype_for_token("string")),
        metadata=result.metadata,
    )


class Host(KernelBase):
    """Only hard-coded invented households; this source accepts no external data."""

    ref = "test.puf_detail.invented_host@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        assert set(context.params) == {"wage_delta"}
        assert context.params["wage_delta"] in (0, 1)
        assert (
            context.sources["literal_host"].read_bytes()
            == b"invented fixed host literals v1\n"
        )
        frame = host(context.params["wage_delta"])
        scope = {
            "scope": detail.SCOPE,
            "native_content_sha256": detail.population_content(frame),
            "source_fixture_sha256": codec.sha(
                codec.encode_json(
                    {
                        "fixture": "literal-two-arm-households-v1",
                        "wage_delta": context.params["wage_delta"],
                    }
                )
            ),
        }
        return KernelResult(
            frame=frame,
            artifacts={"fixture_host": codec.encode_json(scope)},
            receipt={
                "scope": detail.SCOPE,
                "source": "code-owned invented literals only",
            },
        )


class RawScope(KernelBase):
    ref = "test.puf_detail.raw_scope@1"
    capabilities = Host.capabilities

    def run(self, context):
        assert (
            context.sources["literal_scope"].read_bytes()
            == b"invented fixed raw scope v1\n"
        )
        return KernelResult(
            frame=Frame(
                {
                    "person": pd.DataFrame({"person_id": [1], "person_model_id": [1]}),
                    "model": pd.DataFrame({"model_id": [1], "fixture_anchor": [0.0]}),
                },
                EntitySchema(group_entities=("model",)),
                {"model": Weights(np.ones(1), WeightKind.DESIGN)},
            )
        )


def fixture_sources(root, weight_delta=0):
    root.mkdir(exist_ok=True)
    ordinary = [
        source_fixture.main_record(
            recid=str(11 + i),
            mars=str(1 + i % 4),
            s006=str(10 + i + (weight_delta if i == 0 else 0)),
            amounts={
                "E00200": str(10 * i),
                "E00900": str((-9, 0, 17)[i % 3]),
                "E00100": str(4 * i - 11),
            },
        )
        for i in range(16)
    ]
    # Permuted and interleaved aggregates retain delivered order, including decimals.
    aggregates = [
        source_fixture.main_record(
            recid=str(i), mars="0", s006="0", amounts={"E00900": "-1234.50"}
        )
        for i in (999999, 999997, 999996, 999998)
    ]
    records = [
        aggregates[0],
        *ordinary[:8],
        aggregates[1],
        *ordinary[8:],
        *aggregates[2:],
    ]
    decoded, status, projection, definition = source_fixture.build_fixture(records)
    main = source_fixture.csv_bytes(source_fixture.MAIN_HEADER, records)
    demographic = source_fixture.csv_bytes(
        source_fixture.DEMOGRAPHIC_HEADER, source_fixture.DEFAULT_DEMOGRAPHIC
    )
    source_paths = {}
    for name, body in [
        (definition.main.source_name, main),
        (definition.demographic.source_name, demographic),
        ("fixture_cpi", source_fixture.FIXTURE_CPI_RESOURCE_BYTES),
    ]:
        path = root / name
        if path.exists():
            assert path.read_bytes() == body
        else:
            path.write_bytes(body)
        source_paths[name] = path
    status_bytes = raw.encode_return_status(status, definition)
    projection_bytes = agi.encode_agi_projection(
        decoded, projection, definition, status_artifact_sha256=codec.sha(status_bytes)
    )
    payloads = {
        "definition": definition.canonical,
        "projection_definition": projection.canonical,
        "status": status_bytes,
        "projection": projection_bytes,
    }
    inputs = {
        name: {
            "path": str(root / (name + ".fixture")),
            "bytes": len(body),
            "sha256": codec.sha(body),
        }
        for name, body in payloads.items()
    }
    code_head = subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True
    ).strip()
    resource = {
        k: v
        for k, v in source_fixture.FIXTURE_RESOURCE.items()
        if k != "resource_bytes"
    }
    compiled = source_fixture.compile_price_baseline(
        detail.MONEY_FIELDS, **source_fixture.FIXTURE_RESOURCE
    )
    params = {
        "scope": detail.SCOPE,
        "definition": definition.params_text,
        "projection": projection.params_text,
        "resource": codec.encode_json(resource).decode(),
        "inputs": codec.encode_json(inputs).decode(),
        "code_head": code_head,
    }
    return definition, projection, compiled, params, source_paths


class Calls(list):
    def __init__(self):
        super().__init__()
        self.contexts = {}
        self.timings = {}


def setup(root, *, wage_delta=0, weight_delta=0, forbid=False):
    definition, projection, compiled_price, params, source_paths = fixture_sources(
        root / ("source-" + str(weight_delta)), weight_delta
    )
    for name, body in (
        ("literal_scope", b"invented fixed raw scope v1\n"),
        ("literal_host", b"invented fixed host literals v1\n"),
    ):
        path = root / name
        if path.exists():
            assert path.read_bytes() == body
        else:
            path.write_bytes(body)
        source_paths[name] = path
    registry = SourceCodecRegistry()
    raw.register_us_puf_raw_source_codecs(registry, definition=definition)
    # The raw CPI byte codec itself is the real graph source codec.
    registry.register_bytes("raw-bytes-v1", load_raw_bytes)
    store = ContentStore(root / "store", codecs=registry)
    population = host(wage_delta)
    columns = frame_column_declarations(population)
    clone_nodes = clone.us_combined_survey_clone_nodes(
        columns, base="fixture_host", source_channels=("acs", "asec")
    )
    fits = legacy_qrf_train_nodes(
        "fixture_puf_fit",
        population="fixture_puf_donor",
        entity="tax_unit",
        predictors=detail.FEATURES,
        targets=(detail.TARGET,),
        seed=578,
        n_estimators=2,
        zero_atol=0.0,
        phase=detail.SCOPE,
    )
    source_fixture_sha = codec.sha(
        codec.encode_json(
            {"fixture": "literal-two-arm-households-v1", "wage_delta": wage_delta}
        )
    )
    nodes = (
        Node(
            "raw_scope",
            RawScope.ref,
            structural=StructuralDelta.CREATE,
            sources=("literal_scope",),
            outputs=(Owned("model", "fixture_anchor", "float64"),),
        ),
        raw.us_puf_raw_source_node(population="raw_scope", definition=definition),
        agi.us_puf_monetary_agi_source_node(
            population="raw_scope", definition=definition, projection=projection
        ),
        *graph_detail.price_nodes(params, compiled_price),
        Node(
            "fixture_host",
            Host.ref,
            structural=StructuralDelta.CREATE,
            outputs=columns,
            sources=("literal_host",),
            params={"wage_delta": wage_delta},
            artifact_outputs=(ArtifactOutput("fixture_host", graph_detail.HOST_TYPE),),
        ),
        *clone_nodes,
        *fits,
        *graph_detail.host_nodes(
            columns,
            base=clone_nodes[0].id,
            host_producer="fixture_host",
            source_fixture_sha256=source_fixture_sha,
            fit_nodes=fits,
        ),
    )
    graph = Graph(
        "us",
        (
            *raw.us_puf_raw_source_refs(definition),
            SourceRef("fixture_cpi", "raw-bytes-v1"),
            SourceRef("literal_host", "raw-bytes-v1"),
            SourceRef("literal_scope", "raw-bytes-v1"),
        ),
        nodes,
    )
    kernels = KernelRegistry()
    raw.register_us_puf_raw_source_kernels(
        kernels, definition=definition, source_codecs=registry
    )
    agi.register_us_puf_monetary_agi_source_kernels(
        kernels, definition=definition, projection=projection, source_codecs=registry
    )
    growth_graph.register_us_puf_growth_kernels(kernels)
    clone.register_us_combined_survey_clone_kernels(kernels)
    for kernel in (
        Host(),
        RawScope(),
        graph_detail.FixturePriceAdaptKernel(),
        graph_detail.FixturePriceExportKernel(),
        graph_detail.FixtureDonorKernel(),
        graph_detail.DetailBoundaryKernel(),
        graph_detail.DetailMatrixKernel(),
        graph_detail.DetailAttachKernel(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        kernels.register(kernel)
    calls = Calls()
    for instance in kernels.as_mapping().values():
        original = instance.run

        def counted(context, fn=original):
            calls.append(context.node.id)
            calls.contexts[context.node.id] = context
            if forbid:
                raise AssertionError("WARM_KERNEL_CALLED")
            started = time.monotonic()
            try:
                return fn(context)
            finally:
                calls.timings[context.node.id] = time.monotonic() - started

        instance.run = counted
    return compile_graph(graph), store, kernels, source_paths, calls


def execute(root, *, forbid=False, wage_delta=0, weight_delta=0):
    graph, store, kernels, sources, calls = setup(
        root, forbid=forbid, wage_delta=wage_delta, weight_delta=weight_delta
    )
    source_calls = []
    if forbid:

        def no_source(*args, **kwargs):
            source_calls.append("graph source loader")
            raise AssertionError("WARM_SOURCE_CODEC_CALLED")

        store.codecs.load = no_source
        store.codecs.load_bytes = no_source
    captured = {}
    first_boundary = []

    def observe(frame, event, value):
        if (
            event == "return"
            and isinstance(value, Population)
            and value.version in (clone.COMBINED_CLONE_NODE, BOUNDARY)
        ):
            captured[value.version] = {
                k: v.copy() for k, v in value.design_weights.items()
            }
            if (
                value.version == BOUNDARY
                and value.mass_ledger
                and value.mass_ledger[-1].node_id == BOUNDARY
                and not first_boundary
            ):
                first_boundary.append(value)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        manifest = run_graph(
            graph,
            sources=sources,
            store=store,
            kernels=kernels,
            resume="require" if forbid else "auto",
        )
    finally:
        sys.setprofile(previous)
    before, after = first_boundary[0].frame, manifest.population(BOUNDARY)
    assert detail.population_content(before) == detail.population_content(
        manifest.population(clone.COMBINED_CLONE_NODE)
    )
    assert first_boundary[0].mass_ledger[:-1] == manifest.mass_ledger(
        clone.COMBINED_CLONE_NODE
    )
    placement = store.load_bytes(
        manifest.node("fixture_puf_detail.matrix").opaque_artifacts["placement"]
    )
    proof = graph_detail.verify_materialized_transfer(
        before,
        after,
        placement,
        before_design=first_boundary[0].design_weights,
        after_design=captured[BOUNDARY],
        before_ledger=first_boundary[0].mass_ledger,
        after_ledger=manifest.mass_ledger(BOUNDARY),
        boundary_node=BOUNDARY,
    )
    evidence = {
        "scope": detail.SCOPE,
        "nodes": {
            name: {
                "key": receipt.key,
                "hit": receipt.hit,
                "implementation": receipt.kernel_impl_hash,
                "artifacts": dict(receipt.opaque_artifacts),
            }
            for name, receipt in manifest.nodes.items()
        },
        "opaque_payload_sha256": {
            name + "/" + slot: codec.sha(store.load_bytes(key))
            for name, receipt in manifest.nodes.items()
            for slot, key in receipt.opaque_artifacts.items()
        },
        "materialized_content_sha256": detail.population_content(after),
        "kernel_calls": list(calls),
        "kernel_seconds": calls.timings,
        "source_codec_calls_during_required_graph": source_calls,
        "preservation": proof,
        "setup_note": "Code-owned fixture definitions are constructed/decoded before run_graph, including warm setup.",
    }
    (
        root
        / f"graph-evidence-{'require' if forbid else 'auto'}-{wage_delta}-{weight_delta}.json"
    ).write_text(json.dumps(evidence, indent=2) + "\n")
    return manifest, store, calls, captured, proof


def test_price_reader_refuses_unbound_bytes():
    from microcosm.build.us_runtime.puf_detail_transfer import decode_price_arrays

    with pytest.raises(ValueError, match="PRICE"):
        decode_price_arrays(b"not an envelope", expected={}, expected_sha256="0" * 64)


def test_real_invented_transfer(cold):
    manifest, store, calls, _, proof = cold[1]
    assert len(calls) == len(manifest.nodes)
    assert proof["all_nonowned_preserved"]
    after = manifest.population(BOUNDARY)
    assert after.n("household") == 6 and after.n("tax_unit") == 8
    matrix = decode_recipient_matrix(
        store.load_bytes(
            manifest.node("fixture_puf_detail.matrix").opaque_artifacts["matrix"]
        )
    )
    assert matrix.features.iloc[:, 0].tolist() == [10.0, 20.0, 170.0, 30.0]
    assert matrix.features.mars_2.tolist() == [1.0, 0.0, 1.0, 0.0]
    raw_values = codec.read_raw_target(
        store.load_bytes(
            manifest.node("fixture_puf_detail.apply.000").opaque_artifacts["raw_draw"]
        ),
        target=detail.TARGET,
        index=matrix.features.index,
    )
    observed = (
        after.table("tax_unit")
        .set_index("tax_unit_id")
        .loc[matrix.entity_ids, detail.OUTPUT]
        .to_numpy()
    )
    np.testing.assert_array_equal(observed.view("uint64"), raw_values.view("uint64"))
    donor = manifest.population("fixture_puf_donor")
    assert donor.weights_for("tax_unit").kind is WeightKind.DESIGN
    assert len(donor.table("tax_unit")) == 16
    assert set(np.sign(donor.table("tax_unit")[detail.TARGET])) == {-1, 0, 1}


@pytest.fixture(scope="module")
def cold(tmp_path_factory):
    root = tmp_path_factory.mktemp("puf-detail-literals")
    return root, execute(root)


def altered(frame, mutator):
    tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    mutator(tables)
    return Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
    )


def attach_context(cold):
    return cold[1][2].contexts["fixture_puf_detail.attach"]


def change_artifact(context, name, *, payload=None, producer=None):
    from microcosm.graph.keys import opaque_artifact_key

    artifacts = dict(context.artifacts)
    value = artifacts[name]
    if payload is not None:
        value = replace(value, payload=payload)
    if producer is not None:
        edge = next(e for e in context.node.artifact_inputs if e.name == name)
        value = replace(
            value,
            producer_key=producer,
            key=opaque_artifact_key(producer, edge.artifact),
        )
    artifacts[name] = value
    return replace(context, artifacts=artifacts)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("payload", "PRICE_PAYLOAD_SHA"),
        ("body", "PRICE_BODY_SHA"),
        ("trailing", "PRICE_TRAILING_BYTES"),
        ("truncated", "PRICE_BODY_TRUNCATED"),
        ("header", "PRICE_IDENTITY"),
    ],
)
def test_price_reader_refusals(cold, mutation, reason):
    _, (manifest, store, _, _, _) = cold
    receipt = manifest.node("fixture_puf_price.export")
    body = store.load_bytes(receipt.opaque_artifacts["price"])
    binding = codec.decode_json(
        store.load_bytes(receipt.opaque_artifacts["price_binding"])
    )
    expected, sha = dict(binding["header"]), binding["payload_sha256"]
    if mutation in ("payload", "body"):
        body = body[:-1] + bytes([body[-1] ^ 1])
    elif mutation == "trailing":
        body += b"x"
    elif mutation == "truncated":
        body = body[:-1]
    else:
        expected["source_head"] = "f" * 40
    if mutation != "payload":
        sha = codec.sha(body)
    with pytest.raises(ValueError, match=reason):
        detail.decode_price_arrays(body, expected=expected, expected_sha256=sha)


@pytest.mark.parametrize("alias", ["raw_draw", "matrix"])
def test_sibling_producer_refusal(cold, alias):
    context = change_artifact(attach_context(cold), alias, producer="f" * 64)
    with pytest.raises(ValueError, match="DETAIL_SIBLING_PRODUCER"):
        graph_detail.DetailAttachKernel().run(context)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("matrix_key", "DETAIL_MATRIX_PRODUCER"),
        ("seed", "DETAIL_RAW_STATE_BINDING"),
        ("entity", "DETAIL_STATE_ENTITY_INDEX"),
        ("target", "DETAIL_RAW_STATE_BINDING"),
        ("raw_sha", "DETAIL_RAW_STATE_BINDING"),
    ],
)
def test_apply_state_binding(cold, mutation, reason):
    context = attach_context(cold)
    packet = codec.decode_json(context.artifacts["apply_state"].payload)
    if mutation == "matrix_key":
        packet["matrix_producer_key"] = "f" * 64
    elif mutation == "seed":
        packet["application"]["seed"] += 1
    elif mutation == "entity":
        packet["application"]["state"]["entity"] = "household"
    elif mutation == "target":
        packet["application"]["state"]["predictors"] = list(reversed(detail.FEATURES))
    else:
        packet["application"]["raw_targets"][0]["sha256"] = "f" * 64
    context = change_artifact(context, "apply_state", payload=codec.encode_json(packet))
    with pytest.raises(ValueError, match=reason):
        graph_detail.DetailAttachKernel().run(context)


@pytest.mark.parametrize("mutation", ["order", "zero_weight_omission", "feature"])
def test_matrix_exact_selected_rows(cold, mutation):
    from microcosm.fit.model_input import encode_recipient_matrix

    context = attach_context(cold)
    matrix = decode_recipient_matrix(context.artifacts["matrix"].payload)
    features = matrix.features.copy()
    if mutation == "order":
        features = features.iloc[::-1]
    elif mutation == "zero_weight_omission":
        # The second selected unit is the retained zero-weight ASEC household.
        assert context.weights["tax_unit"].values[5] == 0
        features = features.drop(features.index[1])
    else:
        features.iloc[0, 0] += 1
    payload = encode_recipient_matrix(
        features, entity="tax_unit", entity_ids=features.index.to_numpy()
    )
    context = change_artifact(context, "matrix", payload=payload)
    with pytest.raises(ValueError, match="DETAIL_PLACEMENT_BINDING"):
        graph_detail.DetailAttachKernel().run(context)


def test_raw_id_order_even_with_resealed_application(cold):
    context = attach_context(cold)
    matrix = decode_recipient_matrix(context.artifacts["matrix"].payload)
    values = codec.read_raw_target(
        context.artifacts["raw_draw"].payload,
        target=detail.TARGET,
        index=matrix.features.index,
    )
    raw_payload = codec.encode_raw_target(
        values[::-1], target=detail.TARGET, index=matrix.features.index[::-1]
    )
    state = codec.decode_json(context.artifacts["apply_state"].payload)
    state["application"]["raw_targets"][0]["sha256"] = codec.sha(raw_payload)
    context = change_artifact(context, "raw_draw", payload=raw_payload)
    context = change_artifact(context, "apply_state", payload=codec.encode_json(state))
    with pytest.raises(ValueError, match="target/index/order"):
        graph_detail.DetailAttachKernel().run(context)


def test_native_cell_ownership_refused(cold):
    context = attach_context(cold)
    node = replace(context.node, outputs=(Owned("tax_unit", detail.OUTPUT, "float64"),))
    with pytest.raises(ValueError, match="DETAIL_OWNERSHIP"):
        graph_detail.DetailAttachKernel().run(replace(context, node=node))


@pytest.mark.parametrize("kind", ["missing", "complex", "bool"])
def test_missing_or_untyped_host_wage_refuses(cold, kind):
    before = cold[1][0].population(clone.COMBINED_CLONE_NODE)
    name = "asec_reported_wage_income_2024_price"

    def mutate(tables):
        if kind == "missing":
            tables["person"][name] = np.nan
        elif kind == "complex":
            tables["person"][name] = np.ones(len(tables["person"]), dtype=complex) * (
                1 + 2j
            )
        else:
            tables["person"][name] = True

    with pytest.raises(ValueError, match="HOST_MISSING_FEATURE"):
        detail.recipient_matrix(altered(before, mutate))


@pytest.mark.parametrize(
    "kind", ["role_type", "cohort", "source_channel", "membership"]
)
def test_clone_source_and_membership_refusals(cold, kind):
    before = cold[1][0].population(clone.COMBINED_CLONE_NODE)
    p = graph_detail.provenance

    def mutate(tables):
        person = tables["person"]
        role = person[p.support_clone_index_column("person")].eq(1)
        if kind == "role_type":
            person[p.support_clone_index_column("person")] = person[
                p.support_clone_index_column("person")
            ].astype(float)
        elif kind == "cohort":
            person.loc[role, "source_year"] = "2023"
        elif kind == "source_channel":
            # Both candidate wage fields are finite, so lineage, not missing data,
            # must reject this coherent role-1 whole-household source relabel.
            person["asec_reported_wage_income_2024_price"] = 10.0
            person["employment_income_before_lsr"] = 10.0
            for e, table in tables.items():
                selected = table[p.support_clone_index_column(e)].eq(1)
                col = p.support_channel_column(e)
                table.loc[selected, col] = table.loc[selected, col].map(
                    {"asec": "acs", "acs": "asec"}
                )
        else:
            # Move a detail person across two same-role tax units. No weight fit.
            selected = person.index[role]
            person.loc[selected[-4], "person_tax_unit_id"] = person.loc[
                selected[-1], "person_tax_unit_id"
            ]

    reason = (
        "HOST_CLONE_TYPE"
        if kind == "role_type"
        else "HOST_COPIED_SOURCE_CELL"
        if kind != "membership"
        else "HOST_CLONE_MEMBERSHIP"
    )
    with pytest.raises(ValueError, match=reason):
        detail.recipient_matrix(altered(before, mutate))


def test_genuine_source_definition_refused():
    with pytest.raises(ValueError):
        detail.fixture_sources(
            raw.packaged_definition().params_text,
            agi.packaged_agi_projection().params_text,
        )


def test_source_profile_rekeys_only_extension(monkeypatch):
    kernel = graph_detail.DetailMatrixKernel()
    before = kernel.implementation_hash()
    original = raw.csv_acceptance_profile
    monkeypatch.setattr(
        raw,
        "csv_acceptance_profile",
        lambda: {**original(), "invented_test_profile": 1},
    )
    assert kernel.implementation_hash() != before


def test_host_change_reuses_donor_fit(cold):
    root, (original, _, _, _, _) = cold
    changed, _, calls, _, proof = execute(root, wage_delta=1)
    assert (
        changed.node("fixture_puf_fit.000").key
        == original.node("fixture_puf_fit.000").key
    )
    assert changed.node("fixture_puf_fit.000").hit
    assert "fixture_puf_fit.000" not in calls
    for name in (
        "fixture_puf_detail.matrix",
        "fixture_puf_detail.apply.000",
        "fixture_puf_detail.attach",
    ):
        assert (
            changed.node(name).key != original.node(name).key
            and not changed.node(name).hit
        )
    assert proof["all_nonowned_preserved"]


def test_donor_weight_change_rekeys_fit_preserves_host(cold):
    root, (original, _, _, _, _) = cold
    changed, _, calls, _, proof = execute(root, weight_delta=1)
    assert (
        changed.node("fixture_puf_fit.000").key
        != original.node("fixture_puf_fit.000").key
    )
    assert "fixture_puf_fit.000" in calls
    assert (
        changed.node(clone.COMBINED_CLONE_NODE).key
        == original.node(clone.COMBINED_CLONE_NODE).key
    )
    assert changed.node(clone.COMBINED_CLONE_NODE).hit
    np.testing.assert_array_equal(
        changed.population(BOUNDARY).weights_for("household").values.view("uint64"),
        original.population(BOUNDARY).weights_for("household").values.view("uint64"),
    )
    assert proof["all_nonowned_preserved"]


@pytest.mark.parametrize(
    "kind", ["anchor", "zero_anchor", "ledger", "omitted_column", "native_output"]
)
def test_full_materialized_population_refusals(cold, kind):
    _, (manifest, store, _, _, _) = cold
    before, after = (
        manifest.population(clone.COMBINED_CLONE_NODE),
        manifest.population(BOUNDARY),
    )
    placement = store.load_bytes(
        manifest.node("fixture_puf_detail.matrix").opaque_artifacts["placement"]
    )
    ledger = manifest.mass_ledger(BOUNDARY)
    # Separate actual Population fixture: the main pilot has no incoming DESIGN
    # anchor roster, so it cannot claim nonempty-history coverage by itself.
    anchors = np.arange(before.n("household"), dtype="float64")
    first = Population.from_frame(
        before, BOUNDARY, mass_ledger=ledger, design_weights={"household": anchors}
    )
    last = Population.from_frame(
        after, BOUNDARY, mass_ledger=ledger, design_weights={"household": anchors}
    )

    def verify(a, b, payload=placement):
        return graph_detail.verify_materialized_transfer(
            a.frame,
            b.frame,
            payload,
            before_design=a.design_weights,
            after_design=b.design_weights,
            before_ledger=a.mass_ledger,
            after_ledger=b.mass_ledger,
            boundary_node=BOUNDARY,
        )

    assert verify(first, last)["design_anchor_entities"] == 1
    if kind in ("anchor", "zero_anchor"):
        changed = anchors.copy()
        changed[0 if kind == "zero_anchor" else 2] += 1
        last = replace(last, design_weights={"household": changed})
        reason = "MATERIALIZED_DESIGN_CHANGED"
    elif kind == "ledger":
        last = replace(
            last, mass_ledger=(*ledger[:-1], replace(ledger[-1], node_id="forged"))
        )
        reason = "MATERIALIZED_GRAPH_LEDGER_CHANGED"
    elif kind == "omitted_column":
        # A projected kernel cannot see this omitted native source column. The
        # materialized proof rejects its matching projection digest anyway.
        projected = altered(
            before, lambda t: t["person"].drop(columns="age", inplace=True)
        )
        bound = codec.decode_json(placement)
        bound["host_content_sha256"] = detail.population_content(projected)
        placement = codec.encode_json(bound)
        reason = "MATERIALIZED_HOST_BINDING"
    else:
        bad = altered(
            after, lambda t: t["tax_unit"].loc.__setitem__((0, detail.OUTPUT), 123.0)
        )
        last = replace(last, frame=bad)
        reason = "MATERIALIZED_NATIVE_OUTPUT"
    with pytest.raises(ValueError, match=reason):
        verify(first, last, placement)


def test_donor_price_authority_and_original_metadata(cold):
    context = cold[1][2].contexts["fixture_puf_donor"]
    result = graph_detail.FixtureDonorKernel().run(context)
    projection = detail.fixture_sources(
        context.params["definition"], context.params["projection"]
    )[1]
    assert result.receipt["source_projection_document"] == projection.params_text
    assert result.receipt["source_admission"] == detail.SOURCE_ADMISSION
    binding = codec.decode_json(context.artifacts["price_binding"].payload)
    binding["header"]["contract_sha256"] = "f" * 64
    changed = change_artifact(
        context, "price_binding", payload=codec.encode_json(binding)
    )
    with pytest.raises(ValueError, match="DONOR_PRICE_AUTHORITY"):
        graph_detail.FixtureDonorKernel().run(changed)


def test_changed_host_authority_refused(cold):
    context = attach_context(cold)
    scope = codec.decode_json(context.artifacts["fixture_host"].payload)
    scope["scope"] = "genuine"
    changed = change_artifact(context, "fixture_host", payload=codec.encode_json(scope))
    with pytest.raises(ValueError, match="HOST_FIXTURE_AUTHORITY"):
        graph_detail.DetailAttachKernel().run(changed)


@pytest.mark.parametrize(
    "kind",
    ["unknown", "joint_missing_spouse", "survivor_has_spouse", "negative", "nonfinite"],
)
def test_filing_structure_and_wage_refusals(cold, kind):
    before = cold[1][0].population(clone.COMBINED_CLONE_NODE)

    def mutate(tables):
        units = tables["tax_unit"]
        if kind == "unknown":
            units["filing_status_input"] = "UNKNOWN"
        elif kind == "joint_missing_spouse":
            units.loc[
                units.filing_status_input.eq("SURVIVING_SPOUSE"), "filing_status_input"
            ] = "JOINT"
        elif kind == "survivor_has_spouse":
            units.loc[units.filing_status_input.eq("JOINT"), "filing_status_input"] = (
                "SURVIVING_SPOUSE"
            )
        else:
            tables["person"]["asec_reported_wage_income_2024_price"] = (
                -1.0 if kind == "negative" else np.inf
            )

    reason = (
        "HOST_FILING_STATUS"
        if kind == "unknown"
        else "HOST_FILING_STRUCTURE"
        if "spouse" in kind
        else "HOST_NONFINITE_FEATURE"
    )
    with pytest.raises(ValueError, match=reason):
        detail.recipient_matrix(altered(before, mutate))


def test_negative_donor_wage_refuses_before_fit(cold):
    context = cold[1][2].contexts["fixture_puf_donor"]
    _, projection, status, decoded, _, _ = graph_detail.fixture_inputs(context)
    binding = codec.decode_json(context.artifacts["price_binding"].payload)
    arrays = detail.decode_price_arrays(
        context.artifacts["price"].payload,
        expected=binding["header"],
        expected_sha256=binding["payload_sha256"],
    )
    arrays["E00200"] = np.full_like(arrays["E00200"], -1.0)
    with pytest.raises(ValueError, match="DONOR_WAGE_FEATURE"):
        detail.donor_frame(arrays, status, decoded, projection)


def test_materialized_mask_cannot_move_to_person(cold):
    manifest, store, _, anchors, _ = cold[1]
    before = manifest.population(clone.COMBINED_CLONE_NODE)
    after = manifest.population(BOUNDARY)

    def move_mask(tables):
        tables["tax_unit"].drop(columns=detail.MASK, inplace=True)
        tables["person"][detail.MASK] = False

    # This is a valid Frame: the name exists once, on the wrong entity. Merely
    # adding a duplicate name would be rejected by Frame before this boundary.
    moved = altered(after, move_mask)
    placement = store.load_bytes(
        manifest.node("fixture_puf_detail.matrix").opaque_artifacts["placement"]
    )
    ledger = manifest.mass_ledger(BOUNDARY)
    with pytest.raises(ValueError, match="MATERIALIZED_NONOWNED_CHANGED"):
        graph_detail.verify_materialized_transfer(
            before,
            moved,
            placement,
            before_design=anchors[BOUNDARY],
            after_design=anchors[BOUNDARY],
            before_ledger=ledger,
            after_ledger=ledger,
            boundary_node=BOUNDARY,
        )


def test_context_strips_only_tax_unit_owned_names(cold):
    context = attach_context(cold)
    tables = {e: table.copy(deep=True) for e, table in context.tables.items()}
    tables["tax_unit"].drop(columns=detail.MASK, inplace=True)
    tables["person"][detail.MASK] = False
    restored = graph_detail.context_frame(replace(context, tables=tables))
    assert detail.MASK in restored.person and detail.MASK not in restored.table(
        "tax_unit"
    )
