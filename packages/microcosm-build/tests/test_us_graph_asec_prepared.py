"""Tiny real-source/real-engine graph evidence, with cache-only process replay.

All source bytes are invented. Private source pins are substituted only inside
these tests, including the fresh interpreters; no production pin override exists.
"""

import importlib.util
import json
import os
import shutil
import struct
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
from microcosm.build.us_runtime import asec_housing_universe as hu
from microcosm.build.us_runtime import asec_housing_universe_source as hu_source
from microcosm.build.us_runtime import asec_income_observations as income_source
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import asec_prepared_source as prepared
from microcosm.build.us_runtime import graph_asec_income as income
from microcosm.build.us_runtime import graph_asec_prepared as bridge
from microcosm.build.us_runtime import graph_context
from microcosm.build.us_runtime import graph_housing_universe as graph_hu
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
from microcosm.frame import US_SCHEMA, Frame
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
)
from microcosm.graph import (
    ArtifactType,
    ContentStore,
    NodeRejectedError,
    Owned,
    StoreCorrupt,
    StoreMiss,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json

pytestmark = pytest.mark.requires_us

KERNELS = (
    bridge.USAsecPreparedCreateKernel,
    bridge.USAsecPreparedSelectionKernel,
    bridge.USAsecPreparedLeavesKernel,
    bridge.USAsecPreparedEngineKernel,
    income.USAsecReportedIncomeKernel,
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
    income_source._MEMBER_PINS = tuple(tuple(item) for item in pins["income"])


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
        unit_strata=bridge._household_strata(
            frame.person, frame.table("household").household_id.to_numpy(dtype="int64")
        ),
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
    # Independent direct arithmetic oracle over the actual source-bound values.
    paw = income.bind_income_observations(
        value.income_observations_payload, prepared_receipt=value.receipt
    )
    w = selected.field("WSAL_VAL").amounts
    e = selected.field("SEMP_VAL").amounts + selected.field("FRSE_VAL").amounts
    s = selected.field("SSI_VAL").amounts
    p = paw.array("PAW_VAL_2024_price")
    t = selected.field("PTOTVAL").amounts
    for name, amounts in zip(
        income.RESULT_COLUMNS, (w, e, s, p, t, t - (((w + e) + s) + p)), strict=True
    ):
        frame.person[name] = amounts
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
    (root / "direct-housing-universe.bin").write_bytes(value.housing_universe_payload)
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
        _source_fixture().invented_prepared_directory(
            root,
            patch,
            households_per_cohort=2,
            housing_contexts={
                2023: {"H_TENURE": 0},
                2024: {"H_LIVQRT": 10, "HRHTYPE": 10},
            },
        )
        (root / "invented-pins.json").write_bytes(
            canonical_json(
                {
                    "source": legacy._SOURCE_PINS,
                    "members": restoration._MEMBER_PINS,
                    "students": student._MEMBER_PINS,
                    "income": income_source._MEMBER_PINS,
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
        income.REPORTED_INCOME_NODE,
        bridge.ENGINE_NODE,
    ]
    assert not any(record.hit for record in manifest.nodes.values())
    expected = load_frame_checkpoint(root / "direct-frame.h5").frame
    actual = manifest.population(bridge.SELECT_NODE)
    for entity in expected.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity), expected.table(entity), check_exact=True
        )
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


def test_registry_declares_exact_five_real_kernels_and_no_raw_money_leaf_reads(
    executed,
):
    root, *_rest = executed
    graph = _graph(root)
    assert bridge.us_asec_prepared_registry().refs() == tuple(
        sorted(cls.ref for cls in KERNELS)
    )
    assert len(compile_graph(graph).graph.nodes) == 5
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
            replace(selection, params={**selection.params, "fraction": 0.1})
        )


def test_actual_selection_kernel_selects_whole_households_and_preserves_every_axis(
    executed,
):
    """Actual authenticated invented source, nonidentity graph filter and real PE."""
    root, full, store, *_ = executed
    half, half_store = _run(root, resume="auto", fraction=0.5)
    assert half.nodes[bridge.CREATE_NODE].hit
    assert not half.nodes[bridge.SELECT_NODE].hit
    parent, frame = (
        full.population(bridge.CREATE_NODE),
        half.population(bridge.SELECT_NODE),
    )
    assert frame.n("household") == 3 and frame.n("person") == 6
    assert frame.person.groupby(
        "source_year"
    ).person_household_id.nunique().to_dict() == {2022: 1, 2023: 1, 2024: 1}
    ids = frame.table("household").household_id.to_numpy()
    positions = {
        "person": np.flatnonzero(np.isin(parent.person.person_household_id, ids)),
        "household": np.flatnonzero(
            np.isin(parent.table("household").household_id, ids)
        ),
    }
    assert np.array_equal(
        parent.person.person_id.to_numpy()[positions["person"]],
        frame.person.person_id.to_numpy(),
    )
    assert (
        parent.weights_for("household").values[positions["household"]].tobytes()
        == frame.weights_for("household").values.tobytes()
    )
    prep_bytes = _payload(half, half_store, bridge.CREATE_NODE, "prepared_receipt")
    prep = json.loads(prep_bytes)
    body = parse_current_money_body(
        _payload(half, half_store, bridge.CREATE_NODE, "current_money"),
        expected_header_sha256=prep["money_header_sha256"],
        expected_content_sha256=prep["money_content_sha256"],
        field_entities=tuple(tuple(v) for v in prep["field_entities"]),
    )
    selection_bytes = _payload(half, half_store, bridge.SELECT_NODE, "selection")
    selection = json.loads(selection_bytes)
    selected = decode_selected_current_money(
        _payload(half, half_store, bridge.SELECT_NODE, "selected_current_money"),
        expected_parent_header_sha256=body.header_sha256,
        expected_parent_content_sha256=body.content_sha256,
        expected_prepared_receipt_sha256=money._sha(prep_bytes),
        expected_selection_sha256=money._sha(selection_bytes),
    )
    for name, entity in prep["field_entities"]:
        for axis in ("amounts", "statuses", "validity", "zero_origin"):
            assert (
                getattr(selected.field(name), axis).tobytes()
                == getattr(body.field(name), axis)[positions[entity]].tobytes()
            )
    before = decode_engine_evaluation(
        _payload(full, store, bridge.ENGINE_NODE, "engine_evaluation")
    )
    after = decode_engine_evaluation(
        _payload(half, half_store, bridge.ENGINE_NODE, "engine_evaluation")
    )
    for root_name in ADMITTED_ROOTS:
        assert (
            before.values[root_name][positions["person"]].tobytes()
            == after.values[root_name].tobytes()
        )
    assert selection["mass_normalization"] == "none"
    assert selection["mass"]["after"] < selection["mass"]["before"]
    assert (
        frame.table("household").asec_housing_universe_occupied_hu.tolist().count(2)
        == 1
    )
    assert (
        frame.table("household").asec_housing_universe_hu_tenure_class.tolist().count(3)
        == 1
    )


def test_native_hu_bytes_promoted_cells_and_source_lineage_are_separate(executed):
    root, manifest, store, *_ = executed
    prep = json.loads(_payload(manifest, store, bridge.CREATE_NODE, "prepared_receipt"))
    payload = _payload(manifest, store, bridge.CREATE_NODE, "housing_universe")
    assert payload == (root / "direct-housing-universe.bin").read_bytes()
    body = graph_hu.bind_housing_universe(payload, prepared_receipt=prep)
    assert type(body) is graph_hu.BoundHousingUniverse
    assert not isinstance(body, hu.AuthenticatedHousingUniverse)
    frame = manifest.population(bridge.SELECT_NODE)
    household = frame.table("household")
    assert household.asec_housing_universe_occupied_hu.tolist() == [1, 1, 2, 1, 1, 2]
    assert household.asec_housing_universe_hu_tenure_class.tolist() == [
        2,
        3,
        0,
        2,
        3,
        0,
    ]
    for name, alias in zip(hu.DERIVED_COLUMNS, hu_source.ATTACHED_COLUMNS, strict=True):
        assert body.array(name).dtype == np.dtype("uint8")
        assert not body.array(name).flags.writeable
        assert household[alias].dtype == np.dtype("int64")
        assert np.array_equal(household[alias], body.array(name))
    positions = np.asarray([1, 2, 5], dtype="int64")
    # Legitimate subsets remain a row comparison, never a source authority object.
    graph_hu.verify_graph_housing_rows(
        household.iloc[positions], body, positions=positions
    )
    with pytest.raises(hu.HousingUniverseRefusalError, match="EXPECTED_UNIVERSE"):
        hu.decode_housing_universe(payload, expected=body)
    with pytest.raises(hu.HousingUniverseRefusalError, match="AUTHENTICATED_UNIVERSE"):
        hu_source.verify_housing_universe_rows(
            household, body, lineage_ids=household.household_id.to_numpy()
        )
    with pytest.raises(hu.HousingUniverseRefusalError, match="BOUND_HU_CONSTRUCTOR"):
        graph_hu.BoundHousingUniverse(body.header, body.buffers)
    checkpoint = root / "hu-selected-checkpoint.h5"
    write_frame_checkpoint(checkpoint, frame)
    restored = load_frame_checkpoint(checkpoint).frame
    graph_hu.verify_graph_housing_rows(
        restored.table("household"), body, positions=np.arange(6, dtype="int64")
    )
    assert prep["housing_universe"]["parent_kind"] == "HousingStatusAttachedAsec"
    assert (
        prep["pre_housing_universe_frame_sha256"] != prep["pre_promotion_frame_sha256"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "derived",
        "source",
        "definition",
        "codebook",
        "source_claim",
        "synthetic",
        "row_count",
        "schema_bool",
        "codebook_bool",
    ],
)
def test_rechecksummed_hu_transport_cannot_change_native_definition_or_source(
    executed, mutation
):
    _root, manifest, store, *_ = executed
    prep = json.loads(_payload(manifest, store, bridge.CREATE_NODE, "prepared_receipt"))
    payload = _payload(manifest, store, bridge.CREATE_NODE, "housing_universe")
    body = graph_hu.bind_housing_universe(payload, prepared_receipt=prep)
    header, buffers = json.loads(body.header), list(body.buffers)
    if mutation in ("raw", "derived"):
        name = "H_TENURE" if mutation == "raw" else "occupied_hu"
        values = body.array(name).copy()
        values[0] = 1 if mutation == "raw" else 2
        buffers[hu.COLUMNS.index(name)] = values.tobytes()
    elif mutation == "source":
        header["source"]["identity"] = "{}"
    elif mutation == "definition":
        header["definition_sha256"] = "f" * 64
        prep["housing_universe"]["definition_sha256"] = "f" * 64
    elif mutation == "codebook":
        header["codebook"]["occupied_hu"]["occupied_housing_unit"] = 2
    elif mutation == "source_claim":
        header["source"]["income_year_origin"] = "caller"
    elif mutation == "synthetic":
        header["source_authentication"] = "synthetic_unverified"
    elif mutation == "schema_bool":
        header["schema_version"] = True
    elif mutation == "codebook_bool":
        header["codebook"]["occupied_hu"]["occupied_housing_unit"] = True
    else:
        header["household_rows"] += 1
    header_bytes = hu._json(header)
    raw = b"".join(
        (hu.MAGIC, struct.pack("<I", len(header_bytes)), header_bytes, *buffers)
    )
    forged = raw + bytes.fromhex(money._sha(raw))
    prep["housing_universe"].update(
        payload_sha256=money._sha(forged),
        header_sha256=money._sha(header_bytes),
        content_sha256=money._sha(b"microcosm/asec-housing-universe-content/1\0" + raw),
    )
    with pytest.raises(hu.HousingUniverseRefusalError):
        graph_hu.bind_housing_universe(forged, prepared_receipt=prep)


@pytest.mark.parametrize(
    "mutation",
    [
        "negative",
        "large",
        "float",
        "bool",
        "narrow",
        "raw",
        "id_order",
        "positions",
        "producer",
        "type",
        "binding",
    ],
)
def test_selected_hu_cells_and_ancestry_refuse_without_cast_or_authority_upgrade(
    executed, mutation
):
    _root, _manifest, _store, contexts, *_ = executed
    context = contexts[bridge.LEAVES_NODE]
    artifacts, tables = dict(context.artifacts), dict(context.tables)
    household = tables["household"].copy(deep=True)
    alias = "asec_housing_universe_occupied_hu"
    if mutation in ("negative", "large", "raw"):
        col = "asec_H_SEQ" if mutation == "raw" else alias
        household.iloc[0, household.columns.get_loc(col)] = {
            "negative": -1,
            "large": 257,
            "raw": 99,
        }[mutation]
    elif mutation in ("float", "bool", "narrow"):
        household[alias] = household[alias].astype(
            {"float": "float64", "bool": "bool", "narrow": "uint8"}[mutation]
        )
    elif mutation == "id_order":
        household = household.iloc[::-1].reset_index(drop=True)
    elif mutation in ("positions", "binding"):
        value = artifacts["selection"]
        data = json.loads(value.payload)
        if mutation == "positions":
            data["household_positions_sha256"] = "f" * 64
        else:
            data["parent"]["housing_universe_payload_sha256"] = "f" * 64
        artifacts["selection"] = replace(value, payload=canonical_json(data))
    else:
        value = artifacts["housing_universe"]
        artifacts["housing_universe"] = replace(
            value,
            **(
                {"producer_key": "f" * 64}
                if mutation == "producer"
                else {"type": ArtifactType("wrong", 1)}
            ),
        )
    tables["household"] = household
    with pytest.raises(ValueError):
        bridge.USAsecPreparedLeavesKernel().run(
            replace(context, artifacts=artifacts, tables=tables)
        )


def test_full_hu_cohort_disagreement_refuses_before_selection(executed):
    _root, _manifest, _store, contexts, *_ = executed
    context = contexts[bridge.SELECT_NODE]
    tables = dict(context.tables)
    person = tables["person"].copy(deep=True)
    person.loc[person.source_year == 2022, "source_year"] = 2023
    tables["person"] = person
    with pytest.raises(hu.HousingUniverseRefusalError, match="BOUND_HU_COHORTS"):
        bridge.USAsecPreparedSelectionKernel().run(replace(context, tables=tables))


def test_required_cache_refuses_corrupt_hu_bytes_without_execution(
    executed, monkeypatch
):
    root, manifest, store, *_ = executed
    key = manifest.nodes[bridge.CREATE_NODE].opaque_artifacts["housing_universe"]
    path = store.object_path(key) / "payload.bin"
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupt HU cache executed a kernel")

    for cls in KERNELS:
        monkeypatch.setattr(cls, "run", forbidden)
    try:
        path.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
        with pytest.raises((StoreMiss, StoreCorrupt)):
            _run(root, resume="require")
    finally:
        path.write_bytes(before)
