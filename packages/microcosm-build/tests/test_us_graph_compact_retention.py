"""Actual compiler/store/owners over invented financial and completion inputs."""

import gc
import hashlib
import json
import sys
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_us_graph_atomic_completion_host import (
    _assert_all_row_lineage,
    _source_arguments,
)
from test_us_graph_atomic_survey_financial import (
    known_financial_run as known_financial_run,
)

from microcosm.build.us_runtime import graph_atomic_survey_financial as host
from microcosm.build.us_runtime import graph_survey_completion_host as completion
from microcosm.build.us_runtime.survey_population_replay import same_replayed_population


def _same_run(expected, actual):
    assert expected.manifest.key == actual.manifest.key
    assert expected.checked_view().payload == actual.checked_view().payload
    left, right = host._run_entry(expected)[2], host._run_entry(actual)[2]
    assert left.artifact_hashes == right.artifact_hashes
    assert left.node_states == right.node_states
    for name in expected.compiled.order:
        a, b = expected.manifest.node(name), actual.manifest.node(name)
        assert a.key == b.key and a.receipt == b.receipt
    same_replayed_population(expected.financial_population, actual.financial_population)


def _record_snapshots(patch):
    recorder = SimpleNamespace(references=[])
    original = host.run_graph

    def run_graph(compiled, **arguments):
        observer = arguments["_population_observer"]

        def observe(node_id, population):
            recorder.references.append(
                (compiled.order, node_id, weakref.ref(population))
            )
            observer(node_id, population)

        return original(compiled, **{**arguments, "_population_observer": observe})

    patch.setattr(host, "run_graph", run_graph)
    return recorder


def _assert_retention(recorder, runs):
    retained_by_order = {}
    for run in runs:
        state = host._run_entry(run)[2]
        assert state.retention_profile == "compact"
        assert tuple(name for name, _ in state.node_witnesses) == run.compiled.order
        assert len(state.node_populations) == len(state.retained_node_ids)
        retained_by_order[run.compiled.order] = set(state.retained_node_ids)
        if state.completion_boundary is not None:
            boundary = state.completion_boundary
            assert not (set(boundary.observed) & set(boundary.base.compiled.order))
            base_state = host._run_entry(boundary.base)[2]
            assert (
                base_state.retention_profile == "compact"
            )  # recursion did not fall back
            retained_by_order[boundary.base.compiled.order] = set(
                base_state.retained_node_ids
            )
    gc.collect()
    discarded = 0
    for order, node, reference in recorder.references:
        assert order in retained_by_order
        if node not in retained_by_order[order]:
            assert reference() is None
            discarded += 1
        else:
            assert reference() is not None
    assert discarded > 0


def test_financial_compact_cold_required_parity_and_nonretention(known_financial_run):
    case = known_financial_run
    baseline_payload = case.cold.checked_view().payload
    with pytest.MonkeyPatch.context() as patch:
        recorder = _record_snapshots(patch)
        call = {
            **case.call,
            "store_root": case.call["store_root"].parent / "compact-store",
            "_population_retention": "compact",
        }
        cold = host.run_atomic_survey_financial(**call)
        warm = host.run_atomic_survey_financial(**call, resume="require")
        # The wrapper is a live implementation binding; compare while still installed.
        _same_run(cold, warm)
        assert cold.manifest.key == case.cold.manifest.key
        assert cold.checked_view().payload == baseline_payload
        assert all(node.hit for node in warm.manifest.nodes.values())
        _assert_retention(recorder, (cold, warm))
        state = host._run_entry(warm)[2]
        assert state.retained_node_ids == (host.financial.ATTACH_NODE,)
        key = warm.manifest.node(host.financial.APPLY_PREFIX + ".000").opaque_artifacts[
            "raw_draw"
        ]
        directory = warm.store.object_path(key)
        payload_path, metadata_path = directory / "payload.bin", directory / "meta.json"
        payload, metadata = payload_path.read_bytes(), metadata_path.read_bytes()
        changed = payload + b"invented_changed"
        document = json.loads(metadata)
        document["payloads"]["payload.bin"] = {
            "sha256": hashlib.sha256(changed).hexdigest(),
            "size": len(changed),
        }
        try:
            payload_path.write_bytes(changed)
            metadata_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":"))
            )
            assert warm.store.load_bytes(key) == changed
            with pytest.raises(ValueError, match="FINANCIAL_RUN_ARTIFACT_CHANGED$"):
                warm.checked_view()
        finally:
            payload_path.write_bytes(payload)
            metadata_path.write_bytes(metadata)
        original = state.node_witnesses
        object.__setattr__(
            state,
            "node_witnesses",
            ((original[0][0], original[0][1] + b" "), *original[1:]),
        )
        with pytest.raises(ValueError, match="COMPACT_CAPSULE_CHANGED$"):
            warm.checked_view()
        object.__setattr__(state, "node_witnesses", original)
        with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN$"):
            replace(warm).checked_view()


@pytest.mark.parametrize(
    "person_status,household_roles,count",
    [(False, False, 45), (True, False, 49), (True, True, 51)],
)
def test_compact_completion_cold_required_parity_and_live_custody(
    tmp_path, person_status, household_roles, count
):
    with pytest.MonkeyPatch.context() as patch:
        call, _ = _source_arguments(
            tmp_path,
            patch,
            person_status=person_status,
            household_roles=household_roles,
        )
        recorder = _record_snapshots(patch)
        all_cold = host.run_atomic_survey_financial(**call)
        all_warm = host.run_atomic_survey_financial(**call, resume="require")
        # Ignore intentional all-profile retention when checking compact lifetimes.
        recorder.references.clear()
        compact = {
            **call,
            "store_root": tmp_path / "compact-store",
            "_population_retention": "compact",
        }
        cold = host.run_atomic_survey_financial(**compact)
        warm = host.run_atomic_survey_financial(**compact, resume="require")
        for run in (all_warm, cold, warm):
            _same_run(all_cold, run)
            assert len(run.compiled.order) == count
            _assert_all_row_lineage(
                run, household_roles=household_roles, person_status=person_status
            )
        assert all(node.hit for node in warm.manifest.nodes.values())
        _assert_retention(recorder, (cold, warm))
        state = host._run_entry(warm)[2]
        boundary = state.completion_boundary
        assert len(boundary.base.compiled.order) == (39 if person_status else 35)
        assert all(
            warm.manifest.node(node).hit for node in boundary.base.compiled.order
        )
        for node in (
            completion.child.DONOR,
            completion.child.RECIPIENT,
            completion.child.FIT,
            completion.child.DRAW,
        ):
            population = boundary.observed[node]
            assert (
                completion._population_stamp(boundary, warm.compiled, node, population)
                == dict(boundary.observed_stamps)[node]
            )
        node = completion.child.DONOR
        boundary.observed[node] = replace(boundary.observed[node])
        with pytest.raises(ValueError, match="COMPLETION_CUSTODY_CHANGED$"):
            warm.checked_view()
        assert boundary.revoked and boundary.child.revoked


@pytest.mark.parametrize("profile", [True, None, "unknown"])
def test_unknown_retention_profile_refuses_before_source_work(profile):
    with pytest.raises(ValueError, match="RETENTION_PROFILE$"):
        host.run_atomic_survey_financial(
            "never-read",
            snapshot_root="never-created",
            store_root="never-created",
            fraction=None,
            seed=1,
            geography_config=None,
            _population_retention=profile,
        )


@pytest.mark.parametrize(
    "change", ["source", "implementation", "typed_contract", "capabilities", "writers"]
)
def test_frozen_population_inputs_rederive_current_union_obligations(
    known_financial_run, monkeypatch, change
):
    run = known_financial_run.cold
    state = host._run_entry(run)[2]
    populations = dict(
        zip(run.compiled.order, (p for p, _ in state.node_populations), strict=True)
    )
    inputs = {
        node: host.atomic._state_population_inputs(
            run.compiled, node, populations[node]
        )
        for node in run.compiled.order
    }
    receipts = {node: value["receipt"] for node, value in state.node_states.items()}

    def derive():
        return host.atomic._states(
            run.compiled,
            run.kernels,
            dict(state.source_keys),
            {},
            receipts,
            _retained_inputs=inputs,
        )

    baseline = derive()
    host.survey._check_node_states(run.manifest, baseline)
    if change in ("source", "implementation"):
        original = host.atomic._all_node_keys

        def changed(*args):
            keys, implementations = original(*args)
            keys, implementations = dict(keys), dict(implementations)
            target = keys if change == "source" else implementations
            node = run.compiled.order[0]
            target[node] = "a" * 64 if target[node] != "a" * 64 else "b" * 64
            return keys, implementations

        monkeypatch.setattr(host.atomic, "_all_node_keys", changed)
    elif change == "typed_contract":
        monkeypatch.setattr(
            host.atomic,
            "typed_contracts",
            lambda *args: {"invented_changed_scope": True},
        )
    elif change == "capabilities":
        original = host.atomic._capabilities_projection
        monkeypatch.setattr(
            host.atomic,
            "_capabilities_projection",
            lambda value: {**original(value), "invented_changed_scope": True},
        )
    else:
        monkeypatch.setattr(
            host.atomic,
            "_input_writers",
            lambda *args, **kwargs: {
                ("person", "invented"): ("invented_union_writer",)
            },
        )
    current = derive()
    assert current != baseline
    with pytest.raises(ValueError, match="MANIFEST_NODE_STATE$"):
        host.survey._check_node_states(run.manifest, current)


@pytest.mark.parametrize("change", ["node", "version", "mass_partition"])
def test_frozen_population_inputs_refuse_different_compiled_scope(
    known_financial_run, change
):
    from microcosm.graph.canonical import canonical_json

    run = known_financial_run.cold
    state = host._run_entry(run)[2]
    node = run.compiled.order[0]
    population = state.node_populations[0][0]
    payload = json.loads(
        host.atomic._state_population_inputs(run.compiled, node, population)
    )
    payload[change] = "invented_foreign_scope"
    with pytest.raises(ValueError, match="RETAINED_STATE_SCOPE$"):
        host.atomic._states(
            run.compiled,
            run.kernels,
            dict(state.source_keys),
            {},
            {},
            _retained_inputs={node: canonical_json(payload)},
        )


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("duplicate", "ATOMIC_OBSERVER_DUPLICATE"),
        ("missing", "ATOMIC_OBSERVER_ROSTER"),
        ("reordered", "ATOMIC_OBSERVER_ROSTER"),
        ("changed", "SURVEY_POPULATION_REPLAY_"),
    ],
)
def test_compact_discarded_observation_refusals(
    known_financial_run, monkeypatch, defect, reason
):
    original = host.run_graph
    fired = []

    def run_graph(compiled, **arguments):
        observer = arguments["_population_observer"]
        first = []

        def observe(node_id, population):
            if node_id == compiled.order[0]:
                fired.append(True)
                if defect == "missing":
                    return
                if defect == "reordered":
                    first.append((node_id, population))
                    return
                if defect == "duplicate":
                    observer(node_id, population)
                if defect == "changed":
                    table = population.frame.person
                    column = population.frame.schema.entity_id_column(
                        population.frame.schema.person_entity
                    )
                    table.loc[table.index[0], column] += 1
            observer(node_id, population)
            if first:
                observer(*first.pop())

        return original(compiled, **{**arguments, "_population_observer": observe})

    monkeypatch.setattr(host, "run_graph", run_graph)
    with pytest.raises(ValueError, match=reason):
        host.run_atomic_survey_financial(
            **known_financial_run.call,
            resume="require",
            _population_retention="compact",
        )
    assert fired == [True]


def test_compact_expected_population_still_sealed_between_checkpoints(
    known_financial_run,
):
    fired = []

    def profile(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is host.financial.verify_materialized_current_survey_predictors.__code__
            and caller is not None
            and caller.f_code is host.run_atomic_survey_financial.__code__
            and "result" in caller.f_locals
            and not fired
        ):
            fired.append(True)
            population = caller.f_locals["expected"][host.financial.DONOR_COLUMNS_NODE]
            table = population.frame.person
            # An independently reconstructed donor output, not the prepared
            # source Frame (whose earlier source seal would correctly refuse).
            column = host.values.TARGETS[0]
            table.loc[table.index[0], column] += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="ATOMIC_FINAL_POPULATION_MUTATION$"):
            host.run_atomic_survey_financial(
                **known_financial_run.call,
                resume="require",
                _population_retention="compact",
            )
    finally:
        sys.setprofile(previous)
    assert fired == [True]
