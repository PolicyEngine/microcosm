"""Tiny real-source/real-engine graph evidence, with cache-only process replay.

All source bytes are invented. Private source pins are substituted only inside
these tests, including the fresh interpreters; no production pin override exists.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.frame_sampling import sample_frame_households
from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import asec_prepared_source as prepared
from microcosm.build.us_runtime import graph_asec_prepared as bridge
from microcosm.build.us_runtime import graph_context
from microcosm.build.us_runtime.asec_current_money_selection import (
    decode_selected_current_money,
    parse_current_money_body,
    select_current_money,
)
from microcosm.build.us_runtime.asec_engine_evaluation import (
    ADMITTED_ROOTS,
    admit_engine_outputs,
    decode_engine_evaluation,
    materialize_engine_outputs,
)
from microcosm.build.us_runtime.cps_carried_current import (
    CPS_CARRIED_CURRENT_PERSON_LEAVES,
    CPS_CARRIED_CURRENT_ROUTING_COLUMNS,
    CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES,
    derive_cps_carried_current_leaves,
)
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.frame import US_SCHEMA, Frame, Weights
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
)
from microcosm.graph import (
    ArtifactValue,
    ContentStore,
    KernelContext,
    NodeRejectedError,
    Numeric,
    NumericScope,
    Owned,
    StoreMiss,
    compile_graph,
    platform_fingerprint,
    run_graph,
)
from microcosm.graph.canonical import canonical_json

pytestmark = pytest.mark.requires_us

KERNELS = (
    bridge.USAsecPreparedCreateKernel,
    bridge.USAsecPreparedSelectionKernel,
    bridge.USAsecPreparedLeavesKernel,
    bridge.USAsecPreparedEngineKernel,
)
SEED = 578


def _loaded_identity():
    """Record actual worker imports; explicit source-shard runs enforce origins."""
    root = Path(__file__).resolve().parents[3]
    shards = {
        (root / "packages" / f"microcosm-{name}" / "src").resolve()
        for name in ("build", "frame", "graph", "fit", "calibrate", "data")
    }
    configured = {
        Path(path).resolve()
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path
    }
    records = {}
    for name, module in sorted(tuple(sys.modules.items())):
        filename = getattr(module, "__file__", None)
        if not name.startswith("microcosm.") or filename is None:
            continue
        path = Path(filename).resolve()
        if shards <= configured:
            assert any(path.is_relative_to(shard) for shard in shards), (name, path)
        records[name] = {"path": str(path), "sha256": money._sha(path.read_bytes())}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "head": head,
        "modules": records,
        "explicit_six_shards": shards <= configured,
    }


def _source_fixture():
    path = Path(__file__).with_name("test_us_asec_prepared_source.py")
    spec = importlib.util.spec_from_file_location("graph_prepared_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _restore_test_pins(root):
    from microcosm.build.us_runtime import asec_student_controls as student

    pins = json.loads((Path(root) / "invented-pins.json").read_text())
    legacy._SOURCE_PINS = (
        pins["source"][0],
        pins["source"][1],
        tuple(tuple(item) for item in pins["source"][2]),
    )
    restoration._MEMBER_PINS = tuple(tuple(item) for item in pins["members"])
    student._MEMBER_PINS = tuple(tuple(item) for item in pins["students"])


def _direct(root):
    """Call the real preparation, sampler, monetary helper and PE adapter directly."""
    root = Path(root)
    value = prepared.prepare_asec_current_money_population(root / "prepared-source")
    frame = value.frame
    inventory = [
        [item.entity, item.column, item.dtype]
        for item in frame_column_declarations(frame)
    ]
    (root / "columns.json").write_bytes(canonical_json(inventory))
    frame, selection = sample_frame_households(
        frame,
        fraction=1.0,
        seed=SEED,
        source_name=bridge.SELECTION_SOURCE_LABEL,
        unit_strata=np.asarray(["2022", "2023", "2024"], dtype=object),
    )
    body = parse_current_money_body(
        value.money_payload,
        expected_header_sha256=value.receipt["money_header_sha256"],
        expected_content_sha256=value.receipt["money_content_sha256"],
        field_entities=value.field_entities,
    )
    selected = select_current_money(
        body,
        person_positions=np.arange(frame.n("person"), dtype="int64"),
        household_positions=np.arange(frame.n("household"), dtype="int64"),
        prepared_receipt_sha256=value.receipt_sha256,
        selection_sha256=money._sha(canonical_json(selection)),
        person_identity_sha256=graph_context._row_identity(frame.person, "person")[
            "ordered_ids_sha256"
        ],
        household_identity_sha256=graph_context._row_identity(
            frame.table("household"), "household"
        )["ordered_ids_sha256"],
    )
    leaves = derive_cps_carried_current_leaves(
        selected,
        routing=frame.person.loc[:, list(CPS_CARRIED_CURRENT_ROUTING_COLUMNS)],
        spm_membership=frame.person.person_spm_unit_id.to_numpy(dtype="int64"),
        spm_ids=frame.table("spm_unit").spm_unit_id.to_numpy(dtype="int64"),
    )
    for entity, values in (("person", leaves.person), ("spm_unit", leaves.spm_unit)):
        for name in sorted(values):
            frame.table(entity)[name] = values[name]
    write_frame_checkpoint(root / "direct-frame.h5", frame)
    tables = {
        entity: frame.table(entity)
        .loc[
            :,
            [
                US_SCHEMA.entity_id_column(entity),
                *(
                    [
                        US_SCHEMA.membership_column(group)
                        for group in US_SCHEMA.group_entities
                    ]
                    + list(CPS_CARRIED_CURRENT_PERSON_LEAVES)
                    if entity == "person"
                    else list(CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES)
                    if entity == "spm_unit"
                    else []
                ),
            ],
        ]
        .copy()
        for entity in frame.entities
    }
    engine_frame = Frame(
        tables, US_SCHEMA, {"household": frame.weights_for("household")}
    )
    result = materialize_engine_outputs(
        engine_frame,
        PolicyEngineUSEngine(),
        admit_engine_outputs(
            PolicyEngineUSVariableMetadataIndex(),
            produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES,
        ),
    )
    np.savez(root / "direct-outputs.npz", **result.values)
    (root / "direct-header.json").write_bytes(canonical_json(result.header))
    (root / "direct-money.bin").write_bytes(value.money_payload)
    (root / "direct-receipt.json").write_bytes(value.receipt_payload)
    return {
        "pid": os.getpid(),
        "roots": list(result.roots),
        "loaded": _loaded_identity(),
    }


def _graph(root, **params):
    columns = tuple(
        Owned(*entry) for entry in json.loads((Path(root) / "columns.json").read_text())
    )
    return bridge.us_asec_prepared_graph(
        columns,
        fraction=params.pop("fraction", 1.0),
        seed=params.pop("seed", SEED),
        **params,
    )


def _run(root, *, resume="forbid", source_dir=None, **params):
    root = Path(root)
    store = ContentStore(root / "store", codecs=us_source_codecs())
    manifest = run_graph(
        compile_graph(_graph(root, **params)),
        sources={
            bridge.ASEC_PREPARED_SOURCE_NAME: root / "prepared-source"
            if source_dir is None
            else source_dir
        },
        store=store,
        kernels=bridge.us_asec_prepared_registry(),
        resume=resume,
    )
    return manifest, store


def _payload(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


def _summary(manifest, store):
    return {
        "pid": os.getpid(),
        "key": manifest.key,
        "nodes": {name: record.key for name, record in manifest.nodes.items()},
        "artifacts": {
            name: dict(record.opaque_artifacts)
            for name, record in manifest.nodes.items()
        },
        "evaluation_sha256": money._sha(
            _payload(manifest, store, bridge.ENGINE_NODE, "engine_evaluation")
        ),
        "all_hits": all(record.hit for record in manifest.nodes.values()),
        "loaded": _loaded_identity(),
    }


def _cache_only(root):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("cache-only replay executed a kernel or source loader")

    for cls in KERNELS:
        cls.run = forbidden
    prepared.prepare_asec_current_money_population = forbidden
    bridge.prepare_asec_current_money_population = forbidden
    PolicyEngineUSEngine.materialize = forbidden
    manifest, store = _run(root, resume="require")
    result = _summary(manifest, store)
    result["kernel_calls"] = len(calls)
    return result


def _process(root, mode):
    code = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("prepared_graph_process", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._restore_test_pins(sys.argv[2])
result = getattr(module, sys.argv[3])(sys.argv[2])
print(json.dumps(result, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, str(root), mode],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    (Path(root) / f"{mode}.stdout.log").write_text(result.stdout)
    (Path(root) / f"{mode}.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    root = tmp_path_factory.mktemp("invented-prepared-graph")
    from microcosm.build.us_runtime import asec_student_controls as student

    with pytest.MonkeyPatch.context() as patch:
        _source_fixture().invented_prepared_directory(root, patch)
        (root / "invented-pins.json").write_bytes(
            canonical_json(
                {
                    "source": legacy._SOURCE_PINS,
                    "members": restoration._MEMBER_PINS,
                    "students": student._MEMBER_PINS,
                }
            )
        )
        direct = _process(root, "_direct")
        contexts, calls = {}, []
        original_prepare = bridge.prepare_asec_current_money_population

        def counted_prepare(*args, **kwargs):
            calls.append("source_preparation")
            return original_prepare(*args, **kwargs)

        patch.setattr(bridge, "prepare_asec_current_money_population", counted_prepare)
        for cls in KERNELS:
            original = cls.run

            def capture(self, context, *, _run=original):
                contexts.setdefault(context.node.id, context)
                calls.append(context.node.id)
                return _run(self, context)

            patch.setattr(cls, "run", capture)
        manifest, store = _run(root)
        yield root, manifest, store, contexts, calls, direct


def test_real_direct_cold_and_fresh_cache_only_parity(executed):
    root, manifest, store, _contexts, calls, direct = executed
    assert calls == [
        bridge.CREATE_NODE,
        "source_preparation",
        bridge.SELECT_NODE,
        bridge.LEAVES_NODE,
        bridge.ENGINE_NODE,
    ]
    assert not any(record.hit for record in manifest.nodes.values())
    expected = load_frame_checkpoint(root / "direct-frame.h5").frame
    actual = manifest.population(bridge.SELECT_NODE)
    for entity in expected.entities:
        pd.testing.assert_frame_equal(actual.table(entity), expected.table(entity))
        assert (
            actual.resolve_weights(entity).values.tobytes()
            == expected.resolve_weights(entity).values.tobytes()
        )
    evaluation = decode_engine_evaluation(
        _payload(manifest, store, bridge.ENGINE_NODE, "engine_evaluation")
    )
    with np.load(root / "direct-outputs.npz") as oracle:
        for name in ADMITTED_ROOTS:
            assert evaluation.values[name].tobytes() == oracle[name].tobytes()
            assert name not in actual.person.columns
    direct_header = json.loads((root / "direct-header.json").read_text())
    for key in ("aggregates", "closures", "recorded_defaults", "blocked_roots"):
        assert evaluation.header[key] == direct_header[key]
    assert evaluation.header["recorded_defaults"] == []
    assert evaluation.header["release_eligible"] is False
    warm = _process(root, "_cache_only")
    cold = _summary(manifest, store)
    assert len({direct["pid"], warm["pid"], cold["pid"]}) == 3
    for key in ("key", "nodes", "artifacts", "evaluation_sha256"):
        assert warm[key] == cold[key]
    assert warm["all_hits"] and warm["kernel_calls"] == 0


def test_selected_body_preserves_all_33_fields_and_axes_without_readiness(executed):
    root, manifest, store, _contexts, _calls, _direct_result = executed
    receipt = json.loads(
        _payload(manifest, store, bridge.CREATE_NODE, "prepared_receipt")
    )
    body = parse_current_money_body(
        _payload(manifest, store, bridge.CREATE_NODE, "current_money"),
        expected_header_sha256=receipt["money_header_sha256"],
        expected_content_sha256=receipt["money_content_sha256"],
        field_entities=tuple(tuple(item) for item in receipt["field_entities"]),
    )
    selection_payload = _payload(manifest, store, bridge.SELECT_NODE, "selection")
    selection = json.loads(selection_payload)
    selected = decode_selected_current_money(
        _payload(manifest, store, bridge.SELECT_NODE, "selected_current_money"),
        expected_parent_header_sha256=body.header_sha256,
        expected_parent_content_sha256=body.content_sha256,
        expected_prepared_receipt_sha256=money._sha(
            _payload(manifest, store, bridge.CREATE_NODE, "prepared_receipt")
        ),
        expected_selection_sha256=money._sha(selection_payload),
    )
    assert type(selected) is not money.ReadyCurrentMoney
    assert len(selected.fields) == len(money.FIELDS) == 33
    for name in money.FIELDS:
        before, after = body.field(name), selected.field(name)
        assert before == after
    assert selection["mass_normalization"] == "none"
    assert selection["representative"] is False
    assert selection["mass"]["before"] == selection["mass"]["after"]
    assert (root / "direct-money.bin").read_bytes() == _payload(
        manifest, store, bridge.CREATE_NODE, "current_money"
    )
    assert (root / "direct-receipt.json").read_bytes() == _payload(
        manifest, store, bridge.CREATE_NODE, "prepared_receipt"
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("context_ids", "CONTEXT_IDENTITY:person"),
        ("selection_ids", "SELECTION_IDENTITY:person"),
        ("selection_parent", "SELECTION_PARENT_BINDING"),
        ("selected_body", "PAYLOAD_CHECKSUM"),
        ("producer", "ARTIFACT_PRODUCER_SPLIT"),
        ("incumbent", "OWNED_LEAF_INCUMBENT"),
    ],
)
def test_leaf_kernel_refuses_changed_typed_context_selection_body_or_producer(
    executed, mutation, reason
):
    _root, _manifest, _store, contexts, _calls, _direct_result = executed
    context = contexts[bridge.LEAVES_NODE]
    artifacts = dict(context.artifacts)
    if mutation in {"context_ids", "incumbent"}:
        value = artifacts["frame_context"]
        document = json.loads(value.payload)
        if mutation == "context_ids":
            document["entities"]["person"]["ordered_ids_sha256"] = "f" * 64
        else:
            document["entities"]["person"]["columns"].append(
                "employment_income_before_lsr"
            )
        artifacts["frame_context"] = replace(value, payload=canonical_json(document))
    elif mutation in {"selection_ids", "selection_parent"}:
        value = artifacts["selection"]
        document = json.loads(value.payload)
        if mutation == "selection_ids":
            document["entities"]["person"]["ordered_ids_sha256"] = "f" * 64
        else:
            document["parent"]["prepared_receipt_sha256"] = "f" * 64
        artifacts["selection"] = replace(value, payload=canonical_json(document))
    else:
        value = artifacts["selected_current_money"]
        if mutation == "producer":
            artifacts["selected_current_money"] = replace(value, producer_key="f" * 64)
        else:
            payload = bytearray(value.payload)
            payload[-40] ^= 0xFF
            artifacts["selected_current_money"] = replace(value, payload=bytes(payload))
    with pytest.raises(ValueError, match=reason):
        bridge.USAsecPreparedLeavesKernel().run(replace(context, artifacts=artifacts))


def test_registry_declares_exact_four_real_kernels_and_no_raw_money_leaf_reads(
    executed,
):
    root, *_rest = executed
    graph = _graph(root)
    assert bridge.us_asec_prepared_registry().refs() == tuple(
        sorted(cls.ref for cls in KERNELS)
    )
    assert len(compile_graph(graph).graph.nodes) == 4
    leaves = graph.node(bridge.LEAVES_NODE)
    assert leaves.inputs[0].columns == CPS_CARRIED_CURRENT_ROUTING_COLUMNS
    assert not set(leaves.inputs[0].columns) & set(money.FIELDS)
    assert graph.node(bridge.ENGINE_NODE).outputs == ()
    with pytest.raises(bridge.PreparedGraphError, match="OWNED_LEAF_INCUMBENT"):
        bridge.us_asec_prepared_graph(
            (
                *graph.node(bridge.CREATE_NODE).outputs,
                Owned("person", "employment_income_before_lsr", "float64"),
            ),
            fraction=1.0,
            seed=SEED,
        )


@pytest.mark.parametrize(
    "params",
    [
        {"seed": SEED + 1},
        {"fraction": 0.5},
        {"engine_version": "changed"},
        {"period": 2023},
    ],
)
def test_changed_selection_or_model_declaration_cannot_replay_old_cache(
    executed, params
):
    root, *_rest = executed
    with pytest.raises(StoreMiss):
        _run(root, resume="require", **params)


def test_changed_source_bytes_cannot_replay_old_cache_or_receive_authority(executed):
    root, *_rest = executed
    path = root / "prepared-source" / prepared.MEMBER_FILENAMES[0][1]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(StoreMiss):
            _run(root, resume="require")
        with pytest.raises(NodeRejectedError, match="SOURCE_BYTES_MISMATCH"):
            _run(root, resume="forbid")
    finally:
        path.write_bytes(original)


def test_relocated_identical_source_directory_reuses_required_cache(
    executed, tmp_path, monkeypatch
):
    root, cold, _store, _contexts, _calls, _direct_result = executed
    relocated = tmp_path / "renamed-source"
    shutil.copytree(root / "prepared-source", relocated)

    def forbidden(*args, **kwargs):
        raise AssertionError("identical relocated source executed a kernel")

    for cls in KERNELS:
        monkeypatch.setattr(cls, "run", forbidden)
    warm, _store = _run(root, resume="require", source_dir=relocated)
    assert warm.key == cold.key
    assert all(record.hit for record in warm.nodes.values())


def test_required_cache_refuses_actual_runtime_identity_drift_without_kernels(
    executed, monkeypatch
):
    from microcosm.build.us_runtime import asec_engine_evaluation as evaluation

    root, *_rest = executed
    original = evaluation._runtime_package_identity

    def changed(package, module):
        return {**original(package, module), "source_and_parameters_sha256": "f" * 64}

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime drift executed a kernel")

    for cls in KERNELS:
        monkeypatch.setattr(cls, "run", forbidden)
    monkeypatch.setattr(evaluation, "_runtime_package_identity", changed)
    with pytest.raises(ValueError, match="ENGINE_RUNTIME_DRIFT"):
        _run(root, resume="require")


def test_engine_and_selection_refuse_invalid_runtime_params_before_outputs(executed):
    _root, _manifest, _store, contexts, _calls, _direct_result = executed
    engine = contexts[bridge.ENGINE_NODE]
    with pytest.raises(ValueError, match="DECLARED_ENGINE_VERSION"):
        bridge.USAsecPreparedEngineKernel().run(
            replace(engine, params={**engine.params, "engine_version": "changed"})
        )
    selection = contexts[bridge.SELECT_NODE]
    with pytest.raises(ValueError, match="floor"):
        bridge.USAsecPreparedSelectionKernel().run(
            replace(selection, params={**selection.params, "fraction": 0.5})
        )


def test_actual_selection_kernel_selects_whole_households_and_preserves_every_axis(
    executed,
):
    """Two invented households per cohort make the half-sample nontrivial."""
    from microcosm.build.us_runtime import asec_current_money_resources as resources
    from microcosm.build.us_runtime._asec_current_money_codec import (
        current_money_content_sha256,
        encode_current_money,
    )

    _root, manifest, _store, _contexts, _calls, _direct_result = executed
    parent = manifest.population(bridge.CREATE_NODE)
    tables = {}
    for entity in US_SCHEMA.entities:
        original = parent.table(entity)
        extra = original.copy(deep=True)
        extra[US_SCHEMA.entity_id_column(entity)] += 1000
        if entity == "person":
            for group in US_SCHEMA.group_entities:
                extra[US_SCHEMA.membership_column(group)] += 1000
        tables[entity] = pd.concat([original, extra], ignore_index=entity != "person")
    weights = parent.weights_for("household")
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.r_[weights.values, weights.values + 10], weights.kind
            )
        },
        pd.concat([parent.strata, parent.strata]),
    )
    pin = "a" * 64
    source = money.SyntheticAsecSource(
        pin,
        pin,
        ((2022, pin), (2023, pin), (2024, pin)),
        (("ED_VAL.archive", pin), ("ED_VAL.member", pin)),
        pin,
    )
    spec = money.compile_asec_current_money_spec(
        resources.packaged_current_money_resources(
            inspected_version=money.MICROUNIT_VERSION,
            inspected_module_sha256=money.MICROUNIT_SHA256,
        ),
        source,
    )
    person, household = frame.person, frame.table("household")
    scope = money.AsecMoneyScope(
        tuple(person.person_id),
        tuple(household.household_id),
        tuple(person.person_household_id),
        tuple(person.person_spm_unit_id),
        tuple(person.source_year),
        (2022, 2023, 2024, 2022, 2023, 2024),
        tuple(f"p{i}" for i in range(12)),
        tuple(f"h{i}" for i in range(6)),
    )
    # Signed values, NIU annuities, and encoded zeros exercise each field's
    # existing classification; every field is compared at selected positions.
    raw_person = pd.DataFrame(
        {
            field.column: (
                np.full(12, -1.0)
                if field.name == "ANN_VAL"
                else np.tile([123.0, 0.0], 6)
                if field.name == "WSAL_VAL"
                else np.tile([-17.0, 0.0], 6)
                if field.name == "RNT_VAL"
                else np.zeros(12)
            )
            for field in spec.fields
            if field.entity == "person"
        }
    )
    raw_household = pd.DataFrame(
        {
            field.column: np.arange(6, dtype="float64") * 10
            for field in spec.fields
            if field.entity == "household"
        }
    )
    ready = money.require_complete_current_money(
        money.restate_asec_current_money(
            money.classify_asec_money(
                money.AsecMoneyViews(raw_person, raw_household, scope), spec, scope
            ),
            spec,
        ),
        spec,
    )
    receipt = {
        "source_kind": prepared.PREPARED_SOURCE_KIND,
        "release_eligible": False,
        "entity_rows": {entity: frame.n(entity) for entity in frame.entities},
        "money_header_sha256": money._sha(ready.header),
        "money_content_sha256": current_money_content_sha256(ready),
        "field_entities": [[field.name, field.entity] for field in spec.fields],
    }
    graph = bridge.us_asec_prepared_graph(
        frame_column_declarations(frame), fraction=0.5, seed=SEED
    )
    node = graph.node(bridge.SELECT_NODE)
    payloads = {
        "frame_context": graph_context.encode_us_frame_context(frame),
        "prepared_receipt": canonical_json(receipt),
        "current_money": encode_current_money(ready),
    }
    numeric = NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint())
    artifacts = {
        edge.name: ArtifactValue(
            payloads[edge.name],
            edge.type,
            money._sha(payloads[edge.name]),
            pin,
            numeric,
        )
        for edge in node.artifact_inputs
    }
    result = bridge.USAsecPreparedSelectionKernel().run(
        KernelContext(
            node=node,
            tables={
                "person": person.loc[
                    :,
                    [
                        "person_id",
                        *(
                            US_SCHEMA.membership_column(group)
                            for group in US_SCHEMA.group_entities
                        ),
                        "source_year",
                    ],
                ],
                "household": household.loc[
                    :, ["household_id", bridge.HOUSEHOLD_WEIGHT_CARRIER]
                ],
            },
            weights={"household": frame.weights_for("household")},
            strata=frame.strata,
            params=node.params,
            rng=np.random.default_rng(0),
            artifacts=artifacts,
        )
    )
    selection = json.loads(result.artifacts["selection"])
    selected = decode_selected_current_money(
        result.artifacts["selected_current_money"],
        expected_parent_header_sha256=receipt["money_header_sha256"],
        expected_parent_content_sha256=receipt["money_content_sha256"],
        expected_prepared_receipt_sha256=money._sha(payloads["prepared_receipt"]),
        expected_selection_sha256=money._sha(result.artifacts["selection"]),
    )
    keep = result.keep.to_numpy()
    assert keep.sum() == selected.person_rows == 6
    assert selected.household_rows == 3
    ids = np.unique(person.person_household_id.to_numpy()[keep])
    assert np.array_equal(keep, np.isin(person.person_household_id, ids))
    assert person.loc[keep].groupby(
        "source_year"
    ).person_household_id.nunique().to_dict() == {2022: 1, 2023: 1, 2024: 1}
    household_positions = np.flatnonzero(np.isin(household.household_id, ids))
    for name in money.FIELDS:
        before, after = ready.field(name), selected.field(name)
        positions = household_positions if name == "HTOTVAL" else np.flatnonzero(keep)
        for axis in ("amounts", "statuses", "validity", "zero_origin"):
            assert (
                getattr(after, axis).tobytes()
                == getattr(before, axis)[positions].tobytes()
            )
    assert selection["mass_normalization"] == "none"
    assert selection["mass"]["after"] == float(
        frame.resolve_weights("person").values[keep].sum()
    )
    assert selection["mass"]["after"] < selection["mass"]["before"]
    context = json.loads(result.artifacts["frame_context"])
    for entity in US_SCHEMA.group_entities:
        expected_ids = np.unique(person.loc[keep, US_SCHEMA.membership_column(entity)])
        assert context["entities"][entity]["rows"] == len(expected_ids)
        assert (
            context["entities"][entity]["ordered_ids_sha256"]
            == graph_context._row_identity(
                pd.DataFrame({US_SCHEMA.entity_id_column(entity): expected_ids}), entity
            )["ordered_ids_sha256"]
        )
