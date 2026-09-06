"""Mixed-source demographic binding, using invented bytes exclusively."""

import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_current_money_source as legacy
from microcosm.build.us_runtime import asec_demographic_source as demographic
from microcosm.build.us_runtime import asec_prepared_source as prepared
from microcosm.build.us_runtime import graph_composed_asec_binding as arm
from microcosm.build.us_runtime import graph_composed_asec_demographics as binding
from microcosm.build.us_runtime import graph_composed_asec_measures as measures
from microcosm.build.us_runtime import graph_composed_population as composed
from microcosm.build.us_runtime.graph_asec_prepared import PreparedGraphError
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.frame import US_SCHEMA
from microcosm.graph import ContentStore, compile_graph, run_graph
from microcosm.graph.canonical import canonical_json


def _module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _projection(root, patch, *, unknown=False):
    directory = root / "prepared-source"
    source = legacy.load_authenticated_current_money_source(
        directory / prepared.PARENT_FILENAME,
        directory / prepared.HOUSEHOLD_ATTACHMENT_FILENAME,
    )
    members = {}
    pins = []
    for year, name, archive, *_ in demographic._MEMBER_PINS:
        table = pd.read_csv(root / name, dtype=str, keep_default_na=False)
        table["A_SEX"] = [1, 2] * (len(table) // 2)
        table["AXSEX"] = [0, 4] * (len(table) // 2)
        table["A_EXPRRP"] = [1, 5] * (len(table) // 2)
        table["P_SEQ"] = [1, 2] * (len(table) // 2)
        if unknown and year == 2022:
            table.loc[0, "A_SEX"] = 0
            table.loc[1, "A_EXPRRP"] = 0
        path = root / f"demographic-{year}.csv"
        table.iloc[::-1].to_csv(path, index=False)
        members[year] = path
        pins.append(
            (
                year,
                name,
                archive,
                demographic._sha(path.read_bytes()),
                len(table),
                path.stat().st_size,
            )
        )
    patch.setattr(demographic, "_MEMBER_PINS", tuple(pins))
    observed = demographic.load_authenticated_asec_demographic_source(
        source, member_paths=members
    )
    payload = demographic._encode(observed)
    (root / "demographic.bin").write_bytes(payload)
    (root / "demographic-pins.json").write_text(json.dumps(pins))
    return observed


def _graph(root, *, fraction=1.0, projection_path=None):
    payload = (projection_path or root / "demographic.bin").read_bytes()
    projection = binding.admit_asec_demographic_projection(payload)
    columns = tuple(
        __import__("microcosm.graph", fromlist=["Owned"]).Owned(*entry)
        for entry in json.loads((root / "columns.json").read_text())
    )
    return binding.composed_asec_demographic_population_graph(
        columns,
        sample_fraction=fraction,
        sample_seed=17,
        projection_sha256=demographic._sha(payload),
        rows=projection.rows,
        cohort_rows=tuple(
            (int(y), n) for y, n in binding._cohort_rows(projection).items()
        ),
        geography=False,
    )


def _run(
    root, *, fraction=1.0, resume="forbid", store_name="store", projection_path=None
):
    sources = {
        composed.ASEC_PREPARED_SOURCE_NAME: root / "prepared-source",
        composed.ACS_NATIVE_SOURCE_NAME: root / "acs",
        binding.PROJECTION_SOURCE_NAME: projection_path or root / "demographic.bin",
    }
    store = ContentStore(root / store_name, codecs=us_source_codecs())
    registry = binding.composed_asec_demographic_registry(geography=False)
    manifest = run_graph(
        compile_graph(_graph(root, fraction=fraction, projection_path=projection_path)),
        sources=sources,
        store=store,
        kernels=registry,
        resume=resume,
    )
    return manifest, store


def _payload(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


@pytest.fixture(scope="module", params=[False, True], ids=["known", "unknown"])
def executed(tmp_path_factory, request):
    started = time.monotonic()
    root = tmp_path_factory.mktemp("invented-demographic-binding")
    print("PHASE fixture-start", time.monotonic() - started, flush=True)
    with pytest.MonkeyPatch.context() as patch:
        _module("test_us_asec_prepared_source.py").invented_prepared_directory(
            root, patch, households_per_cohort=2
        )
        print("PHASE prepared", time.monotonic() - started, flush=True)
        _module("test_us_graph_composed_population.py").invented_acs_archives(
            root / "acs"
        )
        _module("test_us_graph_composed_asec_binding.py")._write_pins(root)
        direct = composed.compose_from_sources(
            root / "prepared-source", root / "acs", sample_fraction=1.0, sample_seed=17
        )
        (root / "columns.json").write_bytes(
            canonical_json(
                [
                    [x.entity, x.column, x.dtype]
                    for x in frame_column_declarations(direct.frame)
                ]
            )
        )
        print("PHASE direct-composed", time.monotonic() - started, flush=True)
        observed = _projection(root, patch, unknown=request.param)
        print("PHASE projection", time.monotonic() - started, flush=True)
        manifest, store = _run(root)
        print("PHASE graph", time.monotonic() - started, flush=True)
        yield root, manifest, store, observed, request.param


def test_real_graph_binds_exact_source_states_and_preserves_other_cells(executed):
    root, manifest, store, observed, unknown = executed
    frame = manifest.population(composed.HARMONIZE_NODE)
    baseline_graph = measures.composed_asec_population_graph(
        _graph(root).nodes[0].outputs,
        sample_fraction=1.0,
        sample_seed=17,
        geography=False,
    )
    baseline = run_graph(
        compile_graph(baseline_graph),
        sources={
            composed.ASEC_PREPARED_SOURCE_NAME: root / "prepared-source",
            composed.ACS_NATIVE_SOURCE_NAME: root / "acs",
        },
        store=ContentStore(root / "baseline", codecs=us_source_codecs()),
        kernels=measures.composed_asec_population_registry(geography=False),
        resume="forbid",
    )
    before = baseline.population(composed.HARMONIZE_NODE)
    for entity in US_SCHEMA.entities:
        actual = frame.table(entity)
        if entity == "person":
            actual = actual.drop(columns=list(binding.OWNED_COLUMNS))
        expected_table = before.table(entity)
        if entity == "person":
            expected_table = expected_table.drop(
                columns=list(binding.CANONICAL_COLUMNS)
            )
        pd.testing.assert_frame_equal(actual, expected_table, check_exact=True)
    assert frame.weighted_entities == before.weighted_entities
    for entity in frame.weighted_entities:
        assert frame.weights_for(entity).kind == before.weights_for(entity).kind
        np.testing.assert_array_equal(
            frame.weights_for(entity).values, before.weights_for(entity).values
        )
    pd.testing.assert_series_equal(frame.strata, before.strata, check_exact=True)
    person = frame.person
    mask = person[arm.arm_row_column("person")].to_numpy(dtype=bool)
    assert (
        person.loc[
            ~mask,
            [
                cell
                for cell in binding.OWNED_COLUMNS
                if cell not in binding.CANONICAL_COLUMNS
            ],
        ]
        .isna()
        .all()
        .all()
    )
    pd.testing.assert_frame_equal(
        person.loc[~mask, list(binding.CANONICAL_COLUMNS)],
        before.person.loc[~mask, list(binding.CANONICAL_COLUMNS)],
        check_exact=True,
    )
    arm_document = arm.bind_composed_asec_document(
        _payload(manifest, store, arm.BIND_NODE, "asec_binding")
    )
    rows = arm.bind_composed_asec_arm_rows(
        _payload(manifest, store, arm.BIND_NODE, "arm_rows"),
        binding_document=arm_document,
    )
    positions = pd.Index(observed.array("person_id")).get_indexer(
        rows.array("person", "original_ids")
    )
    assert (positions >= 0).all()
    for column, source_column, mapping in [
        (binding.FEMALE_COLUMN, "asec_sex_binding_state", {1: False, 2: True}),
        (
            binding.HOUSEHOLD_HEAD_COLUMN,
            "asec_household_reference_state",
            {1: True, 2: False},
        ),
    ]:
        expected = pd.array(
            [
                mapping.get(int(x), pd.NA)
                for x in observed.array(source_column)[positions]
            ],
            dtype="boolean",
        )
        pd.testing.assert_extension_array_equal(
            person.loc[mask, column].array, expected
        )
    for cell, source_column in binding._RETAINED_STATES:
        np.testing.assert_array_equal(
            person.loc[mask, cell].to_numpy(dtype="int64"),
            observed.array(source_column)[positions],
        )
    document = binding.bind_composed_asec_demographic_document(
        _payload(
            manifest, store, binding.DEMOGRAPHIC_BIND_NODE, "asec_demographic_binding"
        )
    )
    assert document["claims"]["release_eligible"] is False
    assert document["bound"]["totals"]["is_female"]["unknown"] == int(unknown)
    assert document["bound"]["totals"]["is_household_head"]["unknown"] == 2 * int(
        unknown
    )


def test_required_warm_replay_executes_no_kernel(executed, monkeypatch):
    root, manifest, *_ = executed
    # Registry execution is the boundary: every kernel is forbidden on replay.
    registry = binding.composed_asec_demographic_registry(geography=False)
    calls = []
    for item in registry._kernels.values():
        monkeypatch.setattr(
            type(item),
            "run",
            lambda *_a, **_k: (
                calls.append(1) or pytest.fail("warm replay executed a kernel")
            ),
        )
    warm, _ = _run(root, resume="require")
    assert not calls
    assert all(record.hit for record in warm.nodes.values())
    assert {n: r.key for n, r in manifest.nodes.items()} == {
        n: r.key for n, r in warm.nodes.items()
    }


def test_source_subset_sampling_and_reordering_are_identity_based(executed):
    root, manifest, store, *_ = executed
    subset, subset_store = _run(root, fraction=0.5, store_name="subset")
    person = subset.population(composed.HARMONIZE_NODE).person
    mask = person[arm.arm_row_column("person")].to_numpy(dtype=bool)
    assert (
        0
        < mask.sum()
        < manifest.population(composed.HARMONIZE_NODE)
        .person[arm.arm_row_column("person")]
        .sum()
    )
    assert (
        binding.bind_composed_asec_demographic_document(
            _payload(
                subset,
                subset_store,
                binding.DEMOGRAPHIC_BIND_NODE,
                "asec_demographic_binding",
            )
        )["arm"]["projection_is_whole_source"]
        is False
    )


@pytest.mark.parametrize(
    "years,ids,reason",
    [
        ([2024], [999999], "UNRESOLVED"),
        ([2023], [1], "COHORT_MISMATCH"),
        ([2022, 2022], [1, 1], "AMBIGUOUS_ROWS"),
    ],
)
def test_missing_ambiguous_and_cross_cohort_join_refuse(executed, years, ids, reason):
    root, *_ = executed
    projection = binding.admit_asec_demographic_projection(
        (root / "demographic.bin").read_bytes()
    )
    first = projection.array("person_id")[0]
    ids = [int(first) if item == 1 else item for item in ids]
    with pytest.raises(PreparedGraphError, match=f"DEMOGRAPHIC_BINDING_{reason}"):
        binding._resolved_projection_rows(
            projection, income_years=np.asarray(years), person_ids=np.asarray(ids)
        )


@pytest.mark.parametrize("states", [[1, 1], [2, 2], [0, 1], [0, 2]])
def test_one_reference_or_whole_household_unknown(states):
    with pytest.raises(
        PreparedGraphError, match="DEMOGRAPHIC_HOUSEHOLD_REFERENCE_CARDINALITY"
    ):
        binding._household_reference_constraint(np.array([1, 1]), np.asarray(states))
    assert binding._household_reference_constraint(
        np.array([1, 1]), np.array([0, 0])
    ) == (0, 1)
    assert binding._household_reference_constraint(
        np.array([1, 1]), np.array([1, 2])
    ) == (1, 0)


def _rewrite_projection(payload, *, header_change=None, arrays_change=None):
    """Reframe invented bytes, preserving all internal cryptographic checks."""
    value = binding.admit_asec_demographic_projection(payload)
    header = value.header
    arrays = {name: value.array(name).copy() for name in demographic.COLUMNS}
    if arrays_change:
        arrays_change(arrays)
        classification = demographic.classify_asec_demographic_observations(
            income_year=arrays["income_year"],
            household_id=binding._grouping_from_published(
                arrays["income_year"], arrays["source_household_id"]
            ),
            a_sex=arrays["asec_A_SEX"],
            axsex=arrays["asec_AXSEX"],
            a_exprrp=arrays["asec_A_EXPRRP"],
            p_seq=arrays["asec_P_SEQ"],
        )
        arrays.update(classification.states)
        header["diagnostics"] = classification.diagnostics
        header["household_membership"] = binding._recomputed_membership(
            arrays["income_year"], arrays["source_household_id"]
        )
    buffers = [
        np.asarray(arrays[name], dtype="<i8").tobytes() for name in demographic.COLUMNS
    ]
    body = b"".join(buffers)
    header["body_sha256"] = demographic._sha(body)
    header["column_sha256"] = dict(
        zip(demographic.COLUMNS, map(demographic._sha, buffers), strict=True)
    )
    if header_change:
        header_change(header)
    encoded = demographic._json(header)
    prefix = demographic.MAGIC + struct.pack("<I", len(encoded)) + encoded + body
    return prefix + bytes.fromhex(demographic._sha(prefix))


def _resolver_args(executed):
    root, manifest, store, *_ = executed
    projection = binding.admit_asec_demographic_projection(
        (root / "demographic.bin").read_bytes()
    )
    arm_document = arm.bind_composed_asec_document(
        _payload(manifest, store, arm.BIND_NODE, "asec_binding")
    )
    arm_rows = arm.bind_composed_asec_arm_rows(
        _payload(manifest, store, arm.BIND_NODE, "arm_rows"),
        binding_document=arm_document,
    )
    frame = manifest.population(composed.HARMONIZE_NODE)
    mask = frame.person[arm.arm_row_column("person")].to_numpy(dtype=bool)
    person = frame.person.loc[mask].copy(deep=True)
    return person, dict(
        projection=projection,
        binding_document=arm_document,
        arm_rows=arm_rows,
        source_document=binding.bind_projection_document(
            _payload(manifest, store, binding.SOURCE_NODE, "demographic_source")
        ),
        context_document=json.loads(
            _payload(manifest, store, measures.REPORTED_INCOME_NODE, "frame_context")
        ),
        producers={
            "population": manifest.nodes[measures.REPORTED_INCOME_NODE].key,
            "asec_arm_binding": manifest.nodes[arm.BIND_NODE].key,
            "demographic_source": manifest.nodes[binding.SOURCE_NODE].key,
        },
    )


@pytest.mark.parametrize("column", ["source_household_id", "A_LINENO", "A_AGE"])
def test_crosschecks_refuse_before_writing(executed, column):
    person, args = _resolver_args(executed)
    person.iloc[0, person.columns.get_loc(column)] += 1
    with pytest.raises(PreparedGraphError, match=f"DEMOGRAPHIC_CROSSCHECK:{column}"):
        binding.resolve_composed_asec_demographics(person, **args)


def test_direct_binding_equals_graph_document_and_arrays(executed):
    person, args = _resolver_args(executed)
    direct = binding.resolve_composed_asec_demographics(person, **args)
    _root, manifest, store, *_ = executed
    assert direct.payload == _payload(
        manifest, store, binding.DEMOGRAPHIC_BIND_NODE, "asec_demographic_binding"
    )
    for cell in binding.CANONICAL_COLUMNS:
        expected = pd.array(direct.booleans[cell], dtype="boolean")
        expected[direct.unknown[cell]] = pd.NA
        pd.testing.assert_extension_array_equal(person[cell].array, expected)
    assert (
        direct.document["household_membership"][
            "native_household_ids_reused_across_cohorts"
        ]
        == 2
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda h: h["contract"].update(missing_token_policy="fill_zero"), "CONTRACT"),
        (lambda h: h["diagnostics"]["totals"].update(rows=-1), "DIAGNOSTICS"),
        (lambda h: h["household_membership"].update(bijective=False), "MEMBERSHIP"),
        (lambda h: h.update(release_eligible=True), "HEADER_BINDING"),
        (lambda h: h["implementation"].update(schema=99), "IMPLEMENTATION"),
    ],
)
def test_self_consistent_transport_cannot_change_source_semantics(
    executed, change, reason
):
    root, *_ = executed
    payload = _rewrite_projection(
        (root / "demographic.bin").read_bytes(), header_change=change
    )
    with pytest.raises(PreparedGraphError, match=f"DEMOGRAPHIC_PROJECTION_{reason}"):
        binding.admit_asec_demographic_projection(payload)


def test_projection_reordering_does_not_change_join_values(executed):
    root, *_ = executed
    payload = _rewrite_projection(
        (root / "demographic.bin").read_bytes(),
        arrays_change=lambda arrays: arrays.update(
            {k: v[::-1].copy() for k, v in arrays.items()}
        ),
    )
    person, args = _resolver_args(executed)
    original = binding.resolve_composed_asec_demographics(person, **args)
    args["projection"] = binding.admit_asec_demographic_projection(payload)
    args["source_document"] = binding.projection_document(
        args["projection"],
        declared=binding.declared_projection_identity(
            projection_sha256=demographic._sha(payload),
            rows=args["projection"].rows,
            cohort_rows=tuple(
                (int(y), n) for y, n in binding._cohort_rows(args["projection"]).items()
            ),
        ),
    )
    reordered = binding.resolve_composed_asec_demographics(person, **args)
    for cell in binding.CANONICAL_COLUMNS:
        np.testing.assert_array_equal(original.booleans[cell], reordered.booleans[cell])
        np.testing.assert_array_equal(original.unknown[cell], reordered.unknown[cell])


def test_foreign_native_roster_refuses_even_when_numeric_coordinates_match(executed):
    root, manifest, store, *_ = executed
    payload = _rewrite_projection(
        (root / "demographic.bin").read_bytes(),
        header_change=lambda h: h.update(native_person_keys_sha256="0" * 64),
    )
    projection = binding.admit_asec_demographic_projection(payload)
    income_payload = _payload(
        manifest, store, composed.CREATE_NODE, "income_observations"
    )
    income = binding.bind_income_observations(
        income_payload,
        prepared_receipt=json.loads(
            _payload(manifest, store, composed.CREATE_NODE, "prepared_receipt")
        ),
    )
    arm_document = arm.bind_composed_asec_document(
        _payload(manifest, store, arm.BIND_NODE, "asec_binding")
    )
    with pytest.raises(PreparedGraphError, match="DEMOGRAPHIC_NATIVE_ROSTER_IDENTITY"):
        binding.verify_projection_native_roster(
            projection,
            income,
            arm_document,
            income_payload_sha256=demographic._sha(income_payload),
        )


def test_changed_projection_invalidates_only_source_and_binding_keys(executed):
    root, original, *_ = executed
    payload = _rewrite_projection(
        (root / "demographic.bin").read_bytes(),
        arrays_change=lambda arrays: arrays["asec_A_SEX"].__setitem__(0, 2),
    )
    changed_path = root / "demographic-changed.bin"
    changed_path.write_bytes(payload)
    changed, _ = _run(root, resume="auto", projection_path=changed_path)
    affected = {binding.SOURCE_NODE, binding.DEMOGRAPHIC_BIND_NODE}
    assert {n for n, r in changed.nodes.items() if not r.hit} == affected
    assert {
        n for n, r in changed.nodes.items() if r.key != original.nodes[n].key
    } == affected


def test_partial_household_selection_refuses(executed):
    person, args = _resolver_args(executed)
    with pytest.raises(PreparedGraphError, match="DEMOGRAPHIC_INCOMPLETE_HOUSEHOLD"):
        binding._require_complete_households(args["projection"], np.array([0]))


def test_direct_binding_refuses_mismatched_projection_document(executed):
    person, args = _resolver_args(executed)
    args["source_document"]["payload_sha256"] = "0" * 64
    with pytest.raises(PreparedGraphError, match="DEMOGRAPHIC_SOURCE_DOCUMENT_PAYLOAD"):
        binding.resolve_composed_asec_demographics(person, **args)


def _fresh_required_replay(root):
    root = Path(root)
    _module("test_us_graph_composed_population.py")._restore_test_pins(root)
    demographic._MEMBER_PINS = tuple(
        tuple(row) for row in json.loads((root / "demographic-pins.json").read_text())
    )
    registry = binding.composed_asec_demographic_registry(geography=False)
    for kernel in registry._kernels.values():

        def refuse(*_args, **_kwargs):
            raise AssertionError("required fresh replay executed a kernel")

        type(kernel).run = refuse
    manifest, _store = _run(root, resume="require")
    return {
        "pid": os.getpid(),
        "nodes": {n: r.key for n, r in manifest.nodes.items()},
        "all_hits": all(r.hit for r in manifest.nodes.values()),
        "kernel_calls": 0,
        "build_root": str(Path(binding.__file__).resolve()),
    }


def test_fresh_process_required_replay_has_zero_execution(executed):
    root, manifest, *_rest, unknown = executed
    if unknown:
        return  # Unknown nulls are replayed by the in-process test already.
    code = """
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("demographic_replay",sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
print(json.dumps(module._fresh_required_replay(sys.argv[2]),sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, __file__, str(root)],
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PYTHONPYCACHEPREFIX": str(root / "fresh-uncreated-pycache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    (root / "fresh-replay.stdout.log").write_text(result.stdout)
    (root / "fresh-replay.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr[-4000:]
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["pid"] != os.getpid()
    assert receipt["all_hits"] and receipt["kernel_calls"] == 0
    assert receipt["nodes"] == {n: r.key for n, r in manifest.nodes.items()}
    assert receipt["build_root"] == str(Path(binding.__file__).resolve())
