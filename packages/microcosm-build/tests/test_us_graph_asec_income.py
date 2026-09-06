"""Source-blind real preparation/graph accounting; this file never calls PE."""

import csv
import importlib.util
import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_income_observations as income_source
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import asec_prepared_source as prepared
from microcosm.build.us_runtime import asec_student_controls as student
from microcosm.build.us_runtime import graph_asec_income as income
from microcosm.build.us_runtime import graph_asec_prepared as bridge
from microcosm.build.us_runtime.asec_current_money_selection import (
    decode_selected_current_money,
    parse_current_money_body,
)
from microcosm.build.us_runtime.graph_implementation import implementation_hash
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.graph import ContentStore, Owned, compile_graph, run_graph
from microcosm.graph.canonical import canonical_json
from microcosm.graph.errors import StoreMissError

KERNELS = (
    bridge.USAsecPreparedCreateKernel,
    bridge.USAsecPreparedSelectionKernel,
    bridge.USAsecPreparedLeavesKernel,
    income.USAsecReportedIncomeKernel,
)


def _fixture_module():
    spec = importlib.util.spec_from_file_location(
        "reported_income_prepared_fixture",
        Path(__file__).with_name("test_us_asec_prepared_source.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph(columns):
    graph = bridge.us_asec_prepared_graph(columns, fraction=1.0, seed=578)
    # Execute the actual four non-engine branches, preserving their declarations.
    # The separately authorized existing engine fixture checks all five together.
    return replace(
        graph, nodes=tuple(n for n in graph.nodes if n.id != bridge.ENGINE_NODE)
    )


def _run(root, columns, *, resume="forbid"):
    store = ContentStore(root / "store", codecs=us_source_codecs())
    manifest = run_graph(
        compile_graph(_graph(columns)),
        sources={bridge.ASEC_PREPARED_SOURCE_NAME: root / "prepared-source"},
        store=store,
        kernels=bridge.us_asec_prepared_registry(),
        resume=resume,
    )
    return manifest, store


def _payload(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


def _warm_worker(root):
    root = Path(root)
    pins = json.loads((root / "pins.json").read_text())
    legacy._SOURCE_PINS = (
        pins["parent"][0],
        pins["parent"][1],
        tuple(tuple(p) for p in pins["parent"][2]),
    )
    restoration._MEMBER_PINS = tuple(tuple(p) for p in pins["members"])
    student._MEMBER_PINS = restoration._MEMBER_PINS
    income_source._MEMBER_PINS = restoration._MEMBER_PINS

    def forbidden(*args, **kwargs):
        raise AssertionError("WARM_KERNEL_OR_SOURCE_CALL")

    for cls in KERNELS:
        cls.run = forbidden
    bridge.prepare_asec_current_money_population = forbidden
    prepared.prepare_asec_current_money_population = forbidden
    income_source.load_authenticated_income_observations = forbidden
    columns = tuple(
        Owned(*row) for row in json.loads((root / "columns.json").read_text())
    )
    manifest, store = _run(root, columns, resume="require")
    result = {
        "all_hits": all(r.hit for r in manifest.nodes.values()),
        "artifacts": {n: dict(r.opaque_artifacts) for n, r in manifest.nodes.items()},
        "accounting_sha256": money._sha(
            _payload(manifest, store, income.REPORTED_INCOME_NODE, "reported_income")
        ),
        "kernel_calls": 0,
        "source_calls": 0,
        "engine_imported": any(
            n.startswith(("policyengine_us", "policyengine_core")) for n in sys.modules
        ),
    }
    print(json.dumps(result, sort_keys=True))


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    root = tmp_path_factory.mktemp("reported-income-graph")
    with pytest.MonkeyPatch.context() as patch:
        directory = _fixture_module().invented_prepared_directory(
            root,
            patch,
            households_per_cohort=2,
            housing_contexts={
                2023: {"H_TENURE": 0},
                2024: {"H_LIVQRT": 10, "HRHTYPE": 10},
            },
        )
        direct = prepared.prepare_asec_current_money_population(directory)
        columns = frame_column_declarations(direct.frame)
        (root / "pins.json").write_bytes(
            canonical_json(
                {"parent": legacy._SOURCE_PINS, "members": restoration._MEMBER_PINS}
            )
        )
        (root / "columns.json").write_bytes(
            canonical_json([[o.entity, o.column, o.dtype] for o in columns])
        )
        contexts, calls = {}, []
        for cls in KERNELS:
            original = cls.run

            def counted(self, context, _run=original):
                calls.append(context.node.id)
                contexts.setdefault(context.node.id, context)
                return _run(self, context)

            patch.setattr(cls, "run", counted)
        manifest, store = _run(root, columns)
        yield root, direct, columns, contexts, calls, manifest, store


def _inputs(executed):
    root, direct, columns, contexts, calls, manifest, store = executed
    context = contexts[income.REPORTED_INCOME_NODE]
    values = context.artifacts
    receipt = json.loads(values["prepared_receipt"].payload)
    selection = json.loads(values["selection"].payload)
    selected = decode_selected_current_money(
        values["selected_current_money"].payload,
        expected_parent_header_sha256=receipt["money_header_sha256"],
        expected_parent_content_sha256=receipt["money_content_sha256"],
        expected_prepared_receipt_sha256=money._sha(values["prepared_receipt"].payload),
        expected_selection_sha256=money._sha(values["selection"].payload),
    )
    source = income.bind_income_observations(
        values["income_observations"].payload, prepared_receipt=receipt
    )
    kwargs = dict(
        person_ids=context.tables["person"].person_id.to_numpy(),
        income_years=context.tables["person"].source_year.to_numpy(),
        person_positions_sha256=selection["person_positions_sha256"],
        prepared_receipt_sha256=money._sha(values["prepared_receipt"].payload),
        selection_sha256=money._sha(values["selection"].payload),
        selected_money_sha256=money._sha(values["selected_current_money"].payload),
        income_payload_sha256=money._sha(values["income_observations"].payload),
        source_producer_key=values["prepared_receipt"].producer_key,
        selection_producer_key=values["selection"].producer_key,
        frame_context_sha256=money._sha(values["frame_context"].payload),
    )
    return selected, source, kwargs


def test_actual_direct_cold_accounting_axes_and_whole_households(executed):
    root, direct, columns, contexts, calls, manifest, store = executed
    assert len(calls) == 4 and len(set(calls)) == 4
    assert not any(r.hit for r in manifest.nodes.values())
    selected, source, kwargs = _inputs(executed)
    value = income.read_reported_income(
        _payload(manifest, store, income.REPORTED_INCOME_NODE, "reported_income"),
        selected,
        source,
        **kwargs,
    )
    full_money = parse_current_money_body(
        direct.money_payload,
        field_entities=direct.field_entities,
        expected_header_sha256=direct.receipt["money_header_sha256"],
        expected_content_sha256=direct.receipt["money_content_sha256"],
    )
    assert (
        value.array(income.RESULT_COLUMNS[0]).tobytes()
        == full_money.field("WSAL_VAL").amount_bytes
    )
    assert (
        value.array(income.RESULT_COLUMNS[1]).tobytes()
        == (
            full_money.field("SEMP_VAL").amounts + full_money.field("FRSE_VAL").amounts
        ).tobytes()
    )
    assert (value.array(income.RESULT_COLUMNS[1]) < 0).any()
    assert (value.array(income.RESULT_COLUMNS[-1]) < 0).any()
    assert (
        value.array(income.RESULT_COLUMNS[2]).tobytes()
        == full_money.field("SSI_VAL").amount_bytes
    )
    assert (
        value.array(income.RESULT_COLUMNS[4]).tobytes()
        == full_money.field("PTOTVAL").amount_bytes
    )
    for name in income.CONTRIBUTING_FIELDS:
        field = full_money.field(name)
        for suffix, expected in (
            ("status", field.status_bytes),
            ("validity", field.validity_bytes),
            ("zero_origin", field.zero_origin_bytes),
        ):
            assert value.array(name + "." + suffix).tobytes() == expected
    assert 1 in value.array(
        "SSI_VAL.zero_origin"
    )  # frozen-fill origin remains unresolved
    assert 2 in value.array(
        "PTOTVAL.zero_origin"
    )  # authentic T1, not retroactively unresolved
    assert 2 in value.array("PAW_VAL.zero_origin")
    assert len(value.array("person_id")) == 12
    assert (value.array("A_AGE") < 15).sum() == 6
    assert not set(income.RESULT_COLUMNS) & set(direct.frame.person)
    assert (
        _payload(manifest, store, bridge.CREATE_NODE, "housing_universe")
        == direct.housing_universe_payload
    )
    assert (
        _payload(manifest, store, bridge.CREATE_NODE, "income_observations")
        == direct.income_observations_payload
    )
    assert (
        _payload(manifest, store, bridge.CREATE_NODE, "current_money")
        == direct.money_payload
    )
    context = json.loads(
        _payload(manifest, store, income.REPORTED_INCOME_NODE, "frame_context")
    )
    assert set(income.RESULT_COLUMNS) <= set(context["entities"]["person"]["columns"])
    for name in income.ACCOUNTING_COLUMNS:
        assert not value.array(name).flags.writeable


def test_fresh_warm_process_has_equal_artifacts_and_zero_calls(executed):
    root, direct, columns, contexts, calls, manifest, store = executed
    code = """
import importlib.util,sys
def guard(event,args):
    if event == 'import' and args[0].startswith(('policyengine_us','policyengine_core')): raise AssertionError('ENGINE_IMPORT')
    if event in {'socket.connect','socket.bind','socket.sendto','socket.getaddrinfo','socket.gethostbyname'}: raise AssertionError('NETWORK')
    if event == 'open' and isinstance(args[0],str) and args[0].startswith(('/Users/maxghenis/PolicyEngine/_buildo-runtime/','/Users/maxghenis/.cache/microcosm/')): raise AssertionError('GENUINE_SOURCE')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('income_fresh_worker',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module._warm_worker(sys.argv[2])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__)), str(root)],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    (root / "warm.stdout.log").write_text(result.stdout)
    (root / "warm.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    warm = json.loads(result.stdout.splitlines()[-1])
    assert warm["all_hits"] and warm["kernel_calls"] == warm["source_calls"] == 0
    assert warm["engine_imported"] is False
    assert warm["artifacts"] == {
        n: dict(r.opaque_artifacts) for n, r in manifest.nodes.items()
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "source_producer",
        "selected_producer",
        "source_type",
        "source_year",
        "selection_positions",
        "incumbent",
    ],
)
def test_real_kernel_refuses_changed_lineage_and_inputs(executed, mutation):
    context = executed[3][income.REPORTED_INCOME_NODE]
    artifacts, tables = dict(context.artifacts), dict(context.tables)
    if mutation in ("source_producer", "selected_producer"):
        name = (
            "income_observations"
            if mutation == "source_producer"
            else "selected_current_money"
        )
        artifacts[name] = replace(artifacts[name], producer_key="f" * 64)
    elif mutation == "source_type":
        artifacts["income_observations"] = replace(
            artifacts["income_observations"], type=income.US_ASEC_REPORTED_INCOME_TYPE
        )
    elif mutation == "source_year":
        tables["person"] = tables["person"].copy()
        tables["person"]["source_year"] = 2024
    else:
        name = "frame_context" if mutation == "incumbent" else "selection"
        data = json.loads(artifacts[name].payload)
        if mutation == "incumbent":
            data["entities"]["person"]["columns"].append(income.RESULT_COLUMNS[0])
        else:
            data["person_positions_sha256"] = "f" * 64
        artifacts[name] = replace(artifacts[name], payload=canonical_json(data))
    reasons = {
        "source_producer": "ACCOUNTING_PRODUCER|BOUND_INCOME",
        "selected_producer": "ACCOUNTING_PRODUCER|SELECTED_",
        "source_type": "ACCOUNTING_SOURCE_TYPE|INCOME_ARTIFACT_TYPE",
        "source_year": "ACCOUNTING_SOURCE_COHORTS",
        "selection_positions": "ACCOUNTING_SOURCE_POSITIONS|SELECTED_",
        "incumbent": "REPORTED_INCOME_COLUMN|FRAME_CONTEXT",
    }
    with pytest.raises(ValueError, match=reasons[mutation]):
        income.USAsecReportedIncomeKernel().run(
            replace(context, artifacts=artifacts, tables=tables)
        )


def test_derived_reader_reconstructs_instead_of_trusting_checksums(executed):
    selected, source, kwargs = _inputs(executed)
    value = income.derive_reported_income(selected, source, **kwargs)
    payload = bytearray(income.encode_reported_income(value))
    payload[-40] ^= 1
    payload[-32:] = bytes.fromhex(money._sha(payload[:-32]))
    with pytest.raises(money.MoneyRefusalError, match="ACCOUNTING_RECONSTRUCTION"):
        income.read_reported_income(bytes(payload), selected, source, **kwargs)
    changed = {
        **kwargs,
        "person_ids": kwargs["person_ids"][::-1].copy(),
        "income_years": kwargs["income_years"][::-1].copy(),
    }
    with pytest.raises(
        money.MoneyRefusalError,
        match="ACCOUNTING_MONEY_BINDING|ACCOUNTING_SOURCE_POSITIONS",
    ):
        income.derive_reported_income(selected, source, **changed)


@pytest.mark.parametrize(
    "mutation", ["price", "nominal", "source_frame", "scope", "zero_class", "join"]
)
def test_fully_rechecksummed_source_transport_still_requires_its_definition(
    executed, mutation
):
    value = executed[3][income.REPORTED_INCOME_NODE].artifacts["income_observations"]
    receipt = json.loads(
        executed[3][income.REPORTED_INCOME_NODE].artifacts["prepared_receipt"].payload
    )
    offset = len(income_source.MAGIC) + 4
    size = struct.unpack_from("<I", value.payload, len(income_source.MAGIC))[0]
    header = json.loads(value.payload[offset : offset + size])
    body = bytearray(value.payload[offset + size : -32])
    if mutation == "price":
        header["price"]["factors"][0][2] = "000000000000f03f"
    elif mutation == "nominal":
        struct.pack_into("<q", body, 6 * header["rows"] * 8, 1201)
        header["body_sha256"] = money._sha(body)
        header["column_sha256"]["PAW_VAL"] = money._sha(
            body[6 * header["rows"] * 8 : 7 * header["rows"] * 8]
        )
    elif mutation == "source_frame":
        header["source_frame_sha256"] = "f" * 64
    elif mutation == "scope":
        header["scope_sha256"] = "f" * 64
    elif mutation == "zero_class":
        header["zero_origin_derivation"]["cohort_counts"]["2022"][
            "observed_amount"
        ] += 1
    else:
        header["sources"][0]["unreferenced_source_rows"] = 1
    head = money._json(header)
    payload = income_source.MAGIC + struct.pack("<I", len(head)) + head + body
    payload += bytes.fromhex(money._sha(payload))
    receipt["income_observations"].update(
        header_sha256=money._sha(head),
        content_sha256=money._sha(head + body),
        payload_sha256=money._sha(payload),
    )
    reasons = {
        "price": "BOUND_INCOME_DEFINITION",
        "nominal": "BOUND_INCOME_CONTENT",
        "source_frame": "BOUND_INCOME_CONTENT",
        "scope": "BOUND_INCOME_CONTENT",
        "zero_class": "BOUND_INCOME_CONTENT",
        "join": "BOUND_INCOME_CONTENT",
    }
    with pytest.raises(money.MoneyRefusalError, match=reasons[mutation]):
        income.bind_income_observations(payload, prepared_receipt=receipt)


def test_graph_declares_exact_artifacts_generation_and_six_owned_outputs(executed):
    graph = bridge.us_asec_prepared_graph(executed[2], fraction=1.0, seed=578)
    assert len(graph.nodes) == 5
    assert all(n.kernel.endswith("@3") for n in graph.nodes)
    node = graph.node(income.REPORTED_INCOME_NODE)
    assert node.inputs[0].columns == ("source_year",)
    assert node.outputs == income.reported_income_declarations()
    assert [o.name for o in node.artifact_outputs] == [
        "reported_income",
        "frame_context",
    ]
    assert (
        graph.node(bridge.CREATE_NODE).artifact_outputs[-1].type
        == income.US_ASEC_INCOME_OBSERVATIONS_TYPE
    )
    assert bridge.SELECTION_SCHEMA.endswith(".v2")
    for name in income.RESULT_COLUMNS:
        with pytest.raises(bridge.PreparedGraphError, match="OWNED_LEAF_INCUMBENT"):
            bridge.us_asec_prepared_graph(
                (*executed[2], Owned("person", name, "float64")), fraction=1.0, seed=578
            )


def test_live_parser_profile_binds_the_prepared_stage_key_and_warm_reuse(executed):
    """Full graph: a changed csv limit changes the stage key and refuses warm reuse."""
    root, direct, columns, contexts, calls, manifest, store = executed
    limit = csv.field_size_limit()
    baseline = implementation_hash("asec_prepared_v3")
    warm, _ = _run(root, columns, resume="require")
    assert all(r.hit for r in warm.nodes.values())
    try:
        csv.field_size_limit(limit * 2)
        changed = implementation_hash("asec_prepared_v3")
        assert changed != baseline
        with pytest.raises(StoreMissError):
            _run(root, columns, resume="require")
    finally:
        csv.field_size_limit(limit)
    assert implementation_hash("asec_prepared_v3") == baseline
    restored, _ = _run(root, columns, resume="require")
    assert all(r.hit for r in restored.nodes.values())
    assert {n: dict(r.opaque_artifacts) for n, r in restored.nodes.items()} == {
        n: dict(r.opaque_artifacts) for n, r in warm.nodes.items()
    }


def test_prepared_receipt_and_source_header_carry_the_same_parser_profile(executed):
    root, direct, columns, contexts, calls, manifest, store = executed
    profile = income_source.parser_profile()
    assert direct.receipt["income_observations"]["parser_profile"] == profile
    selected, source, kwargs = _inputs(executed)
    header = json.loads(source.header)
    assert header["implementation"]["parser_profile"] == profile
    assert profile["csv.field_size_limit"] == csv.field_size_limit()


def test_contributing_money_fields_are_person_entity(executed):
    selected, source, kwargs = _inputs(executed)
    assert [selected.entity_of(name) for name in income.CONTRIBUTING_FIELDS] == [
        "person"
    ] * 5
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            type(selected), "entity_of", lambda self, name: "household", raising=True
        )
        with pytest.raises(money.MoneyRefusalError, match="ACCOUNTING_FIELD_ENTITY"):
            income.derive_reported_income(selected, source, **kwargs)


def test_hand_computed_paw_and_unallocated_rows_match_bitwise(executed):
    """Independent scalar arithmetic, in the declared W/E/S/P/T/U order."""
    root, direct, columns, contexts, calls, manifest, store = executed
    selected, source, kwargs = _inputs(executed)
    value = income.read_reported_income(
        _payload(manifest, store, income.REPORTED_INCOME_NODE, "reported_income"),
        selected,
        source,
        **kwargs,
    )
    contract = income_source._contract()
    factors = {y: bits for y, _ratio, bits in income_source.price_factors(contract)}
    positions = (
        pd.Index(source.array("person_id"))
        .get_indexer(kwargs["person_ids"])
        .astype("int64")
    )
    nominal = source.array("PAW_VAL")[positions]
    years = source.array("income_year")[positions]
    hand_p, hand_u = [], []
    for row in range(len(nominal)):
        amount = float(nominal[row])
        if int(years[row]) != 2024 and amount != 0.0:
            amount = (
                amount * struct.unpack("<d", bytes.fromhex(factors[int(years[row])]))[0]
            )
        hand_p.append(amount)
        w = float(selected.field("WSAL_VAL").amounts[row])
        e = float(selected.field("SEMP_VAL").amounts[row]) + float(
            selected.field("FRSE_VAL").amounts[row]
        )
        s = float(selected.field("SSI_VAL").amounts[row])
        t = float(selected.field("PTOTVAL").amounts[row])
        hand_u.append(t - (((w + e) + s) + amount))
    expected_p = np.asarray(hand_p, dtype="<f8")
    expected_u = np.asarray(hand_u, dtype="<f8")
    assert value.array(income.RESULT_COLUMNS[3]).tobytes() == expected_p.tobytes()
    assert value.array(income.RESULT_COLUMNS[-1]).tobytes() == expected_u.tobytes()
    assert (expected_u < 0).any()
    assert (value.array(income.RESULT_COLUMNS[1]) < 0).any()
    assert (expected_p > 0).any()
