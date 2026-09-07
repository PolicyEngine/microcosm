"""Invented source-qualified transport and full survey clone consumption.

Control receipts below are visibly invented parser fixtures, not genuine
acceptance evidence. Source/growth/model arithmetic uses the actual kernels.
The external proof may supply the reviewed DEA encoder through a source-only
module import; ordinary tests use the exact same documented binary layout.
"""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_puf_detail_transfer as shared
from microcosm.build.us_runtime import graph_puf_diagnostic_consumer as stage
from microcosm.build.us_runtime import puf_diagnostic_consumer as consumer
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_qrf import LegacyQRFTrainKernel
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactValue,
    ContentStore,
    NumericScope,
    Owned,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import load_raw_bytes
from microcosm.graph.population import Population

detail, codec = shared.detail, shared.codec
CHECKOUT = Path(__file__).resolve().parents[3]


def peer(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(CHECKOUT / "tools"))
import us_survey_lineage_graph as survey  # noqa: E402
from survey_lineage_prep import invented  # noqa: E402


def encode_price(arrays, config, compiled):
    if "reviewed_price_engine" in sys.modules:
        return sys.modules["reviewed_price_engine"].encode_arrays(
            arrays, config, compiled
        )
    header = {
        "schema": "local.puf.price_restatement.arrays.v1",
        "input_binding_sha256": codec.sha(codec.encode_json(config["inputs"])),
        "contract_sha256": compiled.sha256,
        "source_head": consumer.SOURCE_HEAD,
        "helper_head": consumer.HELPER_HEAD,
        "release_eligible": False,
        "columns": array_identity(arrays),
    }
    encoded = codec.encode_json(header)
    return (
        detail.MAGIC
        + len(encoded).to_bytes(8, "big")
        + encoded
        + b"".join(a.tobytes() for _, a in sorted(arrays.items()))
    )


def array_identity(arrays):
    return {
        n: {"dtype": a.dtype.str, "rows": len(a), "sha256": codec.sha(a.tobytes())}
        for n, a in sorted(arrays.items())
    }


def fixture_transport(root):
    """Real source/growth graph, then an invented external receipt chain."""
    old = peer("test_us_puf_detail_transfer")
    graph, store, kernels, sources, calls = old.setup(root)
    export = "fixture_puf_price.export"
    needed = {export}
    todo = [export]
    while todo:
        for node in graph.predecessors[todo.pop()]:
            if node not in needed:
                needed.add(node)
                todo.append(node)
    graph = replace(
        graph.graph, nodes=tuple(n for n in graph.graph.nodes if n.id in needed)
    )
    sources_used = {s for n in graph.nodes for s in n.sources}
    graph = replace(
        graph, sources=tuple(s for s in graph.sources if s.name in sources_used)
    )
    manifest = run_graph(
        compile_graph(graph),
        store=store,
        kernels=kernels,
        sources={n: p for n, p in sources.items() if n in sources_used},
        resume="forbid",
    )
    ctx = calls.contexts[export]
    definition, projection, status, decoded, compiled, _ = shared.fixture_inputs(ctx)
    body = store.load_bytes(manifest.node(export).opaque_artifacts["price"])
    start = len(detail.MAGIC) + 8
    header = codec.decode_json(
        body[start : start + int.from_bytes(body[len(detail.MAGIC) : start], "big")]
    )
    arrays = detail.decode_price_arrays(
        body,
        expected={k: v for k, v in header.items() if k != "columns"},
        expected_sha256=codec.sha(body),
    )
    payloads = {
        "definition": definition.canonical,
        "projection_definition": projection.canonical,
        "status": ctx.artifacts["status"].payload,
        "projection": ctx.artifacts["projection"].payload,
    }
    inputs = {
        n: {"path": "/invented/" + n, **consumer.content_pin(p)}
        for n, p in payloads.items()
    }
    contract = {
        "sha256": compiled.sha256,
        "recipe_sha256": codec.sha(compiled.recipe.identity),
        "factor_table_sha256": compiled.factors.sha256,
        "authority": str(compiled.factors.authority),
        "release_eligible": False,
        "ratio": [313689, 237017],
        "ratio_bits": "0aa9e810012df53f",
        "price_basis": "2015 file baseline restated to 2024 prices",
        "per_record_preliminary_inflation": False,
        "nominal_income_growth_claim": False,
    }
    metadata = shared.plain(decoded.column_metadata)
    bound = {
        "route": "test_fixture",
        "invented": True,
        "ordinary_rows": len(arrays["RECID"]),
        "source_column_metadata": metadata,
        "source_basis_sha256": codec.sha(codec.encode_json(metadata)),
        "contract": contract,
        "runtime": {
            "executable_sha256": "bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6",
            "package_versions": {"numpy": "2.4.6", "pandas": "3.0.3"},
        },
        "source": {
            "head": consumer.SOURCE_HEAD,
            "definition_canonical_sha256": definition.sha256,
            "projection_canonical_sha256": projection.sha256,
        },
    }
    config = {
        "mode": "bound_invented",
        "helper_head": consumer.HELPER_HEAD,
        "source_head": consumer.SOURCE_HEAD,
        "inputs": inputs,
        "bound_source": bound,
        "binding_sha256": codec.sha(codec.encode_json(bound)),
        "ordinary_rows": bound["ordinary_rows"],
    }
    payloads["arrays"] = encode_price(arrays, config, compiled)
    runtime = {
        "source_head": consumer.SOURCE_HEAD,
        "helper_head": consumer.HELPER_HEAD,
        "python": sys.version,
        "executable": sys.executable,
        "executable_sha256": bound["runtime"]["executable_sha256"],
        "versions": bound["runtime"]["package_versions"],
    }
    receipts = {}
    for phase in ("cold", "required"):
        receipts[phase] = {
            "phase": phase,
            "pid": 1 if phase == "cold" else 2,
            "runtime": {
                **runtime,
                "bytecode_isolation": {
                    "isolated": True,
                    "dont_write_bytecode": True,
                    "absent_absolute_prefix": "/invented/absent-" + phase,
                },
            },
            "input_binding": inputs,
            "bound_source_identity": config["binding_sha256"],
            "contract": contract,
            "source_definition_sha256": definition.sha256,
            "projection_definition_sha256": projection.sha256,
            "source_column_metadata_preserved": metadata,
            "array_identity": array_identity(arrays),
            "graph_artifacts": {n: r.key for n, r in manifest.nodes.items()},
            "grown_artifact": consumer.content_pin(payloads["arrays"]),
            "excluded_lexical_artifact": {"bytes": 1, "sha256": codec.sha(b"x")},
            "all_hits": phase == "required",
            "kernel_calls": [] if phase == "required" else list(calls),
            "transform_calls": {"graph_money": 0 if phase == "required" else 1},
            "raising_replay_stubs": phase == "required",
        }
        payloads[phase] = codec.encode_json(receipts[phase])
    payloads["proposal"] = codec.encode_json(
        {
            "config": config,
            "price_execution_authorized": False,
            "release_eligible": False,
        }
    )
    payloads["config"] = codec.encode_json(config)
    payloads["completion"] = codec.encode_json(
        {
            "status": "INVENTED_PRICE_RESTATEMENT_PREPARATION_PASSED",
            "genuine_input_bound": False,
            "bound_source_identity": config["binding_sha256"],
            "runtime": {
                **runtime,
                "bytecode_isolation": {
                    "isolated": True,
                    "dont_write_bytecode": True,
                    "absent_absolute_prefix": "/invented/parent",
                },
            },
            "ordinary_rows": bound["ordinary_rows"],
            "aggregate_rows": 4,
            "config_pin": consumer.content_pin(payloads["config"]),
            **{p: {"sha256": codec.sha(payloads[p])} for p in ("cold", "required")},
        }
    )
    payloads["acceptance"] = codec.encode_json(
        {
            "status": "ACCEPTED_INVENTED_DEVELOPMENTAL_PRICE_RESTATEMENT",
            "source_head": consumer.SOURCE_HEAD,
            "helper_head": consumer.HELPER_HEAD,
            "ordinary_rows": bound["ordinary_rows"],
            "excluded_aggregate_rows": 4,
            "release_eligible": False,
            "fit_admission": detail.SOURCE_ADMISSION,
            "saved_file_pins_checked": [
                consumer.content_pin(p) for p in payloads.values()
            ],
            "reviewed_proposal": consumer.content_pin(payloads["proposal"]),
        }
    )
    return payloads, {n: consumer.content_pin(p) for n, p in payloads.items()}, manifest


def host_fixture(root):
    root.mkdir()
    invented.make(root, CHECKOUT)
    invented.restore(root, CHECKOUT)
    columns = tuple(Owned(*c) for c in json.loads((root / "columns.json").read_text()))
    ident = json.loads((root / "projection-identity.json").read_text())
    ident["cohort_rows"] = tuple(tuple(item) for item in ident["cohort_rows"])
    graph = survey.integrated_graph(
        columns, sample_fraction=1.0, sample_seed=17, **ident
    )
    registry = survey.integrated_registry()
    store = ContentStore(root / "store", codecs=us_source_codecs())
    sources = invented.sources(root)
    manifest = run_graph(
        compile_graph(graph),
        store=store,
        kernels=registry,
        sources=sources,
        resume="forbid",
    )
    edges = stage.host_edges(
        native_binding=survey.NATIVE_BINDING, clone_binding=survey.CLONE_BINDING
    )
    values = {}
    for edge in edges:
        record = manifest.node(edge.producer)
        key = record.opaque_artifacts[edge.artifact]
        values[edge.name] = ArtifactValue(
            payload=store.load_bytes(key),
            type=edge.type,
            key=key,
            producer_key=record.key,
            numerics=NumericScope(),
        )
    pins = {
        n: {
            "producer_key": v.producer_key,
            "artifact_key": v.key,
            "payload_sha256": codec.sha(v.payload),
        }
        for n, v in values.items()
    }
    return graph, registry, store, sources, manifest, values, pins


@pytest.fixture(scope="module")
def source_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("invented-qualified-puf")
    (root / "price").mkdir()
    payloads, pins, price_manifest = fixture_transport(root / "price")
    host = host_fixture(root / "host")
    return root, payloads, pins, price_manifest, host


def test_source_transport_and_host_evidence(source_bundle):
    _, payloads, pins, _, host = source_bundle
    donor, receipt = consumer.qualify_price_transport(
        payloads, pins, route="test_fixture", recipe=consumer.recipe_document()
    )
    assert donor.n("tax_unit") == 16
    assert (donor.table("tax_unit").E00900 < 0).any()
    assert (donor.table("tax_unit").E00900 == 0).any()
    assert (
        codec.decode_json(receipt)["original_source_admission"]
        == detail.SOURCE_ADMISSION
    )
    frame = host[4].population(survey.CLONE_POPULATION)
    evidence = consumer.qualify_host_population(frame, host[5])
    assert evidence["all_clone_rows_including_zero_weights"]
    assert evidence["wage_evidence"]["asec"]["evidence_axes"]["validity"]
    assert False in evidence["wage_evidence"]["acs"]["observed"]


@pytest.mark.parametrize("role", consumer.ROLES)
def test_price_changed_payload_refuses(source_bundle, role):
    _, payloads, pins, _, _ = source_bundle
    changed = {**payloads, role: payloads[role] + b"x"}
    with pytest.raises(ValueError, match="PRICE_TRANSPORT_PIN"):
        consumer.qualify_price_transport(
            changed, pins, route="test_fixture", recipe=consumer.recipe_document()
        )


def test_fixture_never_grants_genuine_route(source_bundle):
    _, payloads, pins, _, _ = source_bundle
    with pytest.raises(ValueError, match="PRICE_TRANSPORT_AUTHORITY_SCOPE"):
        consumer.qualify_price_transport(
            payloads, pins, route="packaged", recipe=consumer.recipe_document()
        )


@pytest.mark.parametrize("kind", ("wage", "age", "cohort", "omit_source_column"))
def test_full_host_binding_refuses(source_bundle, kind):
    host = source_bundle[-1]
    frame = host[4].population(survey.CLONE_POPULATION)
    tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    person = tables["person"]
    if kind == "omit_source_column":
        tables["person"] = person.drop(columns=["WAGP"])
    else:
        name = {
            "wage": "employment_income_before_lsr",
            "age": "age",
            "cohort": "source_year",
        }[kind]
        person.loc[person.index[0], name] = 2030 if kind == "cohort" else 99999.0
    changed = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
    )
    with pytest.raises(
        ValueError,
        match="HOUSEHOLD_COHORT_AMBIGUOUS"
        if kind == "cohort"
        else "HOST_FULL_ORIGIN_BINDING",
    ):
        consumer.qualify_host_population(changed, host[5])


def test_missing_acs_earner_never_age_zero():
    person = pd.DataFrame(
        {
            "person_id": [1],
            "AGEP": [14],
            "age": [14.0],
            "source_year": ["2024"],
            "WAGP": [np.nan],
            "ADJINC": [1e6],
            "employment_income_before_lsr": [np.nan],
        }
    )
    proof = consumer.acs_wage_evidence(person)
    assert proof["observed"] == [False] and proof["in_universe"] == [False]
    person["employment_income_before_lsr"] = 0.0
    with pytest.raises(ValueError, match="ACS_WAGE_ADJUSTED_BINDING"):
        consumer.acs_wage_evidence(person)


def diagnostic_setup(root):
    invented.restore(root / "host", CHECKOUT)
    meta = json.loads((root / "diagnostic-inputs.json").read_text())
    identity = json.loads((root / "host/projection-identity.json").read_text())
    identity["cohort_rows"] = tuple(tuple(item) for item in identity["cohort_rows"])
    graph = survey.integrated_graph(
        tuple(Owned(*c) for c in json.loads((root / "host/columns.json").read_text())),
        sample_fraction=1.0,
        sample_seed=17,
        **identity,
    )
    extra = stage.diagnostic_nodes(
        tuple(Owned(*c) for c in meta["columns"]),
        base=survey.CLONE_POPULATION,
        native_binding=survey.NATIVE_BINDING,
        clone_binding=survey.CLONE_BINDING,
        price_pins=meta["price_pins"],
        host_pins=meta["host_pins"],
        route="test_fixture",
    )
    graph = replace(
        graph, sources=(*graph.sources, *stage.SOURCES), nodes=(*graph.nodes, *extra)
    )
    registry = survey.integrated_registry()
    stage.register_diagnostic_kernels(registry)
    for kernel in (
        shared.DetailBoundaryKernel(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        registry.register(kernel)
    codecs = us_source_codecs()
    codecs.register_bytes("raw-bytes-v1", load_raw_bytes)
    store = ContentStore(root / "host/store", codecs=codecs)
    sources = {
        **invented.sources(root / "host"),
        **{
            name: root / "transport" / (role + ".bin")
            for role, name in stage.SOURCE_NAMES.items()
        },
    }
    return compile_graph(graph), registry, store, sources


def execute(root, *, forbid=False):
    """Actual six-node diagnostic append; fresh required replay reads no source codec."""
    graph, registry, store, sources = diagnostic_setup(root)
    calls = []
    contexts = {}
    source_calls = []
    for kernel in registry.as_mapping().values():
        fn = kernel.run

        def counted(context, fn=fn):
            calls.append(context.node.id)
            contexts[context.node.id] = context
            if forbid:
                raise AssertionError("REQUIRED_KERNEL_CALLED")
            return fn(context)

        kernel.run = counted
    if forbid:

        def no_source(*args, **kwargs):
            source_calls.append(1)
            raise AssertionError("REQUIRED_SOURCE_CODEC_CALLED")

        store.codecs.load = no_source
        store.codecs.load_bytes = no_source
    snapshots = []
    original = Population.__post_init__
    boundary = stage.PREFIX + ".population"

    def observe(self):
        original(self)
        if self.version == boundary:
            snapshots.append(self)

    Population.__post_init__ = observe
    try:
        manifest = run_graph(
            graph,
            store=store,
            kernels=registry,
            sources=sources,
            resume="require" if forbid else "auto",
        )
    finally:
        Population.__post_init__ = original
    assert snapshots
    before, after = snapshots[0], snapshots[-1]
    placement = store.load_bytes(
        manifest.node(stage.PREFIX + ".matrix").opaque_artifacts["placement"]
    )
    preservation = shared.verify_materialized_transfer(
        before.frame,
        after.frame,
        placement,
        before_design=before.design_weights,
        after_design=after.design_weights,
        before_ledger=before.mass_ledger,
        after_ledger=after.mass_ledger,
        boundary_node=boundary,
    )
    evidence = {
        "scope": consumer.SCOPE,
        "nodes": {n: {"key": r.key, "hit": r.hit} for n, r in manifest.nodes.items()},
        "kernel_calls": calls,
        "source_codec_calls_during_required_graph": source_calls,
        "opaque_payload_sha256": {
            n + "/" + a: codec.sha(store.load_bytes(k))
            for n, r in manifest.nodes.items()
            for a, k in r.opaque_artifacts.items()
        },
        "materialized_content_sha256": detail.population_content(after.frame),
        "preservation": preservation,
        "setup_note": "Only saved invented declaration/pin metadata is parsed before required run_graph; no price/source numeric decoding or fit in setup.",
    }
    (
        root / ("diagnostic-required.json" if forbid else "diagnostic-cold.json")
    ).write_text(json.dumps(evidence, indent=2) + "\n")
    return manifest, store, calls, contexts, preservation


def test_actual_diagnostic_append(source_bundle):
    root, payloads, pins, _, host = source_bundle
    # Bounded reader and complete feature qualification precede fit explicitly.
    consumer.qualify_price_transport(
        payloads, pins, route="test_fixture", recipe=consumer.recipe_document()
    )
    frame = host[4].population(survey.CLONE_POPULATION)
    consumer.qualify_host_population(frame, host[5])
    (root / "transport").mkdir()
    for n, payload in payloads.items():
        (root / "transport" / (n + ".bin")).write_bytes(payload)
    (root / "diagnostic-inputs.json").write_text(
        json.dumps(
            {
                "price_pins": pins,
                "host_pins": host[6],
                "columns": [
                    [c.entity, c.column, c.dtype]
                    for c in frame_column_declarations(frame)
                ],
            }
        )
    )
    manifest, store, calls, contexts, proof = execute(root)
    assert len(manifest.nodes) == 23 and len(calls) == 6
    assert sum(r.hit for r in manifest.nodes.values()) == 17
    assert proof["all_nonowned_preserved"] and proof["scope"] == consumer.SCOPE
    draw = contexts[stage.PREFIX + ".attach"].artifacts["raw_draw"].payload
    matrix = contexts[stage.PREFIX + ".attach"].artifacts["matrix"].payload
    prepared = shared.model_input.decode_recipient_matrix(matrix)
    values = codec.read_raw_target(
        draw, target=detail.TARGET, index=prepared.features.index
    )
    result = (
        manifest.population(stage.PREFIX + ".population")
        .table("tax_unit")
        .set_index("tax_unit_id")
    )
    assert np.array_equal(
        result.loc[prepared.entity_ids, detail.OUTPUT].to_numpy().view("uint64"),
        values.view("uint64"),
    )


@pytest.mark.parametrize("kind", ("unknown", "null", "duplicate_head", "joint_spouse"))
def test_actual_source_role_refusals(source_bundle, kind):
    frame = source_bundle[-1][4].population(survey.CLONE_POPULATION)
    tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    person, units = tables["person"], tables["tax_unit"]
    role = "tax_unit_role_input"
    source_id = detail.origin.support_source_id_column("person")
    if kind in ("unknown", "null"):
        selected = person[source_id].eq(person[source_id].iloc[0])
        person.loc[selected, role] = "UNSUPPORTED" if kind == "unknown" else None
    else:
        joint = units.loc[units.filing_status_input.eq("JOINT")].iloc[0]
        spouse = person.loc[
            person.person_tax_unit_id.eq(joint.tax_unit_id) & person[role].eq("SPOUSE")
        ].iloc[0]
        selected = person[source_id].eq(spouse[source_id])
        person.loc[selected, role] = "HEAD" if kind == "duplicate_head" else "DEPENDENT"
    changed = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
    )
    with pytest.raises(
        ValueError,
        match="HOST_ROLE_INPUT_VALUES"
        if kind in ("unknown", "null")
        else "HOST_FILING_STRUCTURE",
    ):
        consumer.recipient_matrix(changed)


@pytest.mark.parametrize(
    "kind",
    ("cohort", "age", "wage", "native_id", "reported_payload", "prepared_producer"),
)
def test_asec_actual_evidence_refusals(source_bundle, kind):
    host = source_bundle[-1]
    frame = host[4].population(survey.CLONE_POPULATION)
    person = (
        frame.person.loc[
            frame.person.person_support_clone_index.eq(0)
            & frame.person.person_support_channel.eq("asec")
        ]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    values = dict(host[5])
    if kind == "reported_payload":
        old = values["reported_income"]
        values["reported_income"] = replace(
            old, payload=old.payload[:-1] + bytes([old.payload[-1] ^ 1])
        )
        reason = "ACCOUNTING_RECONSTRUCTION"
    elif kind == "prepared_producer":
        values["prepared_receipt"] = replace(
            values["prepared_receipt"], producer_key="f" * 64
        )
        reason = "BINDING_SOURCE_PRODUCER"
    else:
        column, value, reason = {
            "cohort": ("source_year", 2024, "ACCOUNTING_SOURCE_COHORTS"),
            "age": ("A_AGE", 99, "ASEC_HOST_NATIVE_AGE"),
            "wage": (
                "asec_reported_wage_income_2024_price",
                123.0,
                "ASEC_HOST_WAGE_BINDING",
            ),
            "native_id": (
                "person_spine_source_id",
                999999,
                "ASEC_HOST_NATIVE_PERSON_ROSTER",
            ),
        }[kind]
        person.loc[0, column] = value
    with pytest.raises(ValueError, match=reason):
        consumer.asec_wage_evidence(person, values)
