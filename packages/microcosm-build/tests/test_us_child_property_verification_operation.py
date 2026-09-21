"""Genuine invented source/graph proofs for operation-scoped reconstruction reuse.

No country engine or actual survey files. Exact function-entry counters leave
existing profile hooks active; no source owner or verifier is replaced.
"""

import shutil
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import pytest
import test_us_child_property_income_graph_owner as graph_fixture
import test_us_child_property_income_source_owner as source_fixture

from microcosm.build.us_runtime import graph_child_property_income as graph


class _GenuineDiskPatch:
    def __init__(self, patch):
        self.patch = patch
        self.real_disk_usage = shutil.disk_usage
        self.suppressed = 0

    def setattr(self, target, name, value, *args, **kwargs):
        if target is shutil and name == "disk_usage":
            assert shutil.disk_usage is self.real_disk_usage
            self.suppressed += 1
            return
        self.patch.setattr(target, name, value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.patch, name)


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    root = tmp_path_factory.mktemp("child-verification-operation")
    with pytest.MonkeyPatch.context() as patch:
        genuine = _GenuineDiskPatch(patch)
        _, partial_args, _ = source_fixture.source_arguments(root, genuine)
        assert genuine.suppressed > 0
        assert shutil.disk_usage is genuine.real_disk_usage
        partial = graph.child.source.prepare_authenticated_survey_population(
            **partial_args
        )
        yield SimpleNamespace(partial=partial)
        partial._checked()


@pytest.fixture(autouse=True)
def _case_source_verification_epoch(request):
    """Reuse the production source scope within one genuine owner case only."""
    if "actual" not in request.fixturenames:
        yield
        return
    source = graph.child.source
    assert source.epoch_record() is None
    with source.verification_epoch() as record:
        yield
    # Pytest preserves body failures independently of generator teardown.
    # Teardown still performs the real complete source finalization.
    assert source.epoch_record() is None
    assert record["misses"] >= 1
    assert record["hits"] >= 1
    assert record["final_validations"] >= 1


TOOL = 5
TOOL_NAME = "microcosm-exact-call-counter"
MAX_TARGETS = 32
MAX_CALLS = 1_000_000


def require(condition, reason):
    if not condition:
        raise ValueError("CALL_COUNTER_" + reason)


@contextmanager
def count_calls(targets):
    """Count current-thread starts of unique ordinary Python functions only.

    A free fixed tool ID is required. IDs 3 and 4 belong to the guardian and
    phase recorder. Other tools and existing profile hooks keep running.
    """
    require(sys.version_info >= (3, 13), "PYTHON_MINIMUM")
    require(0 < len(targets) <= MAX_TARGETS, "TARGET_COUNT")
    codes = {}
    for label, function in targets.items():
        require(type(label) is str and bool(label), "LABEL")
        require(type(function) is FunctionType, "FUNCTION")
        code = function.__code__
        require(not code.co_flags & (0x20 | 0x80 | 0x200), "GENERATOR")
        require(code not in codes, "DUPLICATE_CODE")
        codes[code] = label
    require(sys.monitoring.get_tool(TOOL) is None, "TOOL_OCCUPIED")
    owner_thread = threading.get_ident()
    original_profile = sys.getprofile()
    original_thread_profile = threading.getprofile()
    other_tools = tuple(
        (i, sys.monitoring.get_tool(i), sys.monitoring.get_events(i))
        for i in range(6)
        if i != TOOL
    )
    counts = Counter()
    total = 0

    def started(code, _offset):
        nonlocal total
        if threading.get_ident() != owner_thread:
            return
        require(total < MAX_CALLS, "CALL_LIMIT")
        total += 1
        counts[codes[code]] += 1

    installed_codes = []
    sys.monitoring.use_tool_id(TOOL, TOOL_NAME)
    try:
        require(sys.monitoring.get_events(TOOL) == 0, "GLOBAL_EVENTS")
        require(
            sys.monitoring.register_callback(
                TOOL, sys.monitoring.events.PY_START, started
            )
            is None,
            "CALLBACK_OCCUPIED",
        )
        for code in codes:
            sys.monitoring.set_local_events(TOOL, code, sys.monitoring.events.PY_START)
            installed_codes.append(code)
        try:
            yield counts
        finally:
            require(threading.get_ident() == owner_thread, "THREAD_CHANGED")
            require(sys.getprofile() is original_profile, "PROFILE_CHANGED")
            require(
                threading.getprofile() is original_thread_profile,
                "THREAD_PROFILE_CHANGED",
            )
            require(
                other_tools
                == tuple(
                    (i, sys.monitoring.get_tool(i), sys.monitoring.get_events(i))
                    for i in range(6)
                    if i != TOOL
                ),
                "OTHER_TOOL_CHANGED",
            )
            require(sys.monitoring.get_tool(TOOL) == TOOL_NAME, "TOOL_CHANGED")
            require(sys.monitoring.get_events(TOOL) == 0, "GLOBAL_EVENTS_CHANGED")
            require(
                all(
                    sys.monitoring.get_local_events(TOOL, code)
                    == sys.monitoring.events.PY_START
                    for code in codes
                ),
                "LOCAL_EVENTS_CHANGED",
            )
    finally:
        # Do not clear a foreign owner if task code replaced this tool.
        if sys.monitoring.get_tool(TOOL) == TOOL_NAME:
            try:
                for code in installed_codes:
                    sys.monitoring.set_local_events(TOOL, code, 0)
            finally:
                sys.monitoring.clear_tool_id(TOOL)
                sys.monitoring.free_tool_id(TOOL)


def _calls(*, include_proofs=False):
    targets = {
        "reconstruct": graph._reconstruct,
        "current": graph.ChildPropertyBoundary._current,
    }
    if include_proofs:
        targets["proof"] = graph._MaterializedProof.check
    return count_calls(targets)


def test_first_and_closing_reconstructions_replace_only_repeated_middle_work(
    actual, tmp_path
):
    result = graph_fixture._run(actual, tmp_path)
    # The fixture really executes cold and required on the same store. Required
    # hits skip kernels, but its owner verifier already ran independently.
    assert all(item.hit for item in result.warm.nodes.values())
    expected = result.check(result.warm, result.captured)
    before = graph.child.physical._population_stamp(result.captured[graph.VERIFY])
    with _calls(include_proofs=True) as calls:
        with graph.verification_operation():
            for _ in range(4):
                assert result.check(result.warm, result.captured) == expected
            assert calls == {"reconstruct": 1, "current": 8}
        # Earlier proof seals are scanned once at close, not quadratically on
        # each middle borrow. Each call above supplies fresh evidence mappings.
        assert calls == {"reconstruct": 2, "current": 10, "proof": 4}
    assert (
        graph.child.physical._population_stamp(result.captured[graph.VERIFY]) == before
    )
    assert graph._VERIFICATION_OPERATION.get() is None


def test_outside_operation_and_separate_operations_do_not_share_success(
    actual, tmp_path
):
    result = graph_fixture._run(actual, tmp_path)
    with _calls() as calls:
        for _ in range(3):
            result.check(result.warm, result.captured)
        assert calls == {"reconstruct": 3, "current": 6}
        for _ in range(2):
            with graph.verification_operation():
                result.check(result.warm, result.captured)
                with graph.verification_operation():
                    result.check(result.warm, result.captured)
        assert calls == {"reconstruct": 7, "current": 18}


@pytest.mark.parametrize("surface", ["expected", "donor", "payload"])
def test_mutated_retained_reconstruction_is_refused(actual, tmp_path, surface):
    result = graph_fixture._run(actual, tmp_path)
    with pytest.raises(ValueError, match="OPERATION_RECONSTRUCTION_CHANGED"):
        with graph.verification_operation():
            result.check(result.warm, result.captured)
            memo = graph._VERIFICATION_OPERATION.get().entries[result.boundary]
            expected, payloads, donor, _ = memo.value
            if surface == "expected":
                expected.frame.person.loc[0, "unrelated_amount"] += 1
            elif surface == "donor":
                donor.iloc[0, donor.columns.get_loc(graph.WEIGHT)] += 1
            else:
                payloads[graph.VERIFY, "verification"] += b" "
            result.check(result.warm, result.captured)
    assert result.boundary.revoked
    assert graph._VERIFICATION_OPERATION.get() is None


@pytest.mark.parametrize(
    "when", ["warm_first", "warm_final", "close_first", "close_final"]
)
@pytest.mark.parametrize("surface", ["population", "artifact"])
def test_callbacks_still_refuse_mutations_during_warm_and_closing_checks(
    actual, tmp_path, when, surface
):
    result = graph_fixture._run(actual, tmp_path)
    inputs = result.evidence(result.warm, result.captured)
    mutated = []
    try:
        with pytest.raises(ValueError):
            with graph.verification_operation():
                result.boundary.verify_materialized(**inputs)
                target = result.calls.count + (1 if when.endswith("first") else 2)

                def mutate(count):
                    if count != target:
                        return
                    mutated.append(surface)
                    if surface == "population":
                        inputs["population"].frame.person.loc[
                            0, "unrelated_amount"
                        ] += 1
                    else:
                        value = inputs["artifacts"]["draws"]
                        inputs["artifacts"]["draws"] = replace(
                            value, payload=value.payload + b" "
                        )

                result.calls.on_call = mutate
                if when.startswith("warm"):
                    result.boundary.verify_materialized(**inputs)
        assert mutated == [surface]
        assert result.boundary.revoked
        assert graph._VERIFICATION_OPERATION.get() is None
    finally:
        result.calls.on_call = None


def test_close_checks_earlier_evidence_after_last_participant_callback(
    actual, tmp_path
):
    result = graph_fixture._run(actual, tmp_path)
    first = result.evidence(result.warm, result.captured)
    last = result.evidence(result.warm, result.captured)
    try:
        with pytest.raises(ValueError, match="OPERATION_EVIDENCE_CHANGED"):
            with graph.verification_operation():
                result.boundary.verify_materialized(**first)
                result.boundary.verify_materialized(**last)
                target = result.calls.count + 2

                def mutate(count):
                    if count == target:
                        value = first["artifacts"]["draws"]
                        first["artifacts"]["draws"] = replace(
                            value, payload=value.payload + b" "
                        )

                result.calls.on_call = mutate
        assert result.boundary.revoked
    finally:
        result.calls.on_call = None


@pytest.mark.parametrize(
    "surface", ["qualified", "origins", "options", "source_pin", "function", "issuer"]
)
def test_warm_scope_does_not_bless_changed_inputs_or_implementation(
    actual, tmp_path, surface
):
    result = graph_fixture._run(actual, tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        with pytest.raises(ValueError):
            with graph.verification_operation():
                result.check(result.warm, result.captured)
                boundary = result.boundary
                if surface == "qualified":
                    donor = boundary.qualified.donor_projection.donors
                    donor.iloc[0, donor.columns.get_loc(graph.WEIGHT)] += 1
                elif surface == "origins":
                    boundary.origins.iloc[0, 0] = "changed"
                elif surface == "options":
                    boundary.options = replace(boundary.options, scenario_id="changed")
                elif surface == "source_pin":
                    patch.setattr(graph, "_SOURCE_BYTES", ())
                elif surface == "issuer":
                    patch.setitem(
                        graph.child.source._ISSUED, id(boundary.preparation), None
                    )
                else:
                    patch.setattr(graph, "_reconstruct", lambda *args: None)
                result.check(result.warm, result.captured)
        assert result.boundary.revoked
    assert graph._VERIFICATION_OPERATION.get() is None


def test_caught_nested_failure_poisoning_prevents_reuse_or_scope_restart(
    actual, tmp_path
):
    result = graph_fixture._run(actual, tmp_path)
    with pytest.raises(ValueError, match="VERIFICATION_OPERATION_INACTIVE"):
        with graph.verification_operation():
            result.check(result.warm, result.captured)
            with pytest.raises(RuntimeError, match="invented"):
                with graph.verification_operation():
                    raise RuntimeError("invented nested failure")
    assert result.boundary.revoked
    assert graph._VERIFICATION_OPERATION.get() is None
    with pytest.raises(ValueError, match="REVOKED"):
        with graph.verification_operation():
            result.check(result.warm, result.captured)


def test_later_participant_close_cannot_mutate_earlier_verified_result(
    actual, tmp_path
):
    result = graph_fixture._run(actual, tmp_path)
    first_boundary = result.boundary
    second_boundary = graph.ChildPropertyBoundary(
        first_boundary.preparation,
        first_boundary.parent,
        require_parent=first_boundary.require_parent,
        options=first_boundary.options,
        host_edges=first_boundary.host_edges,
        host_pins=first_boundary.host_pins,
    )
    inputs = result.evidence(result.warm, result.captured)
    try:
        with pytest.raises(ValueError, match="OPERATION_RECONSTRUCTION_CHANGED"):
            with graph.verification_operation():
                first_boundary.verify_materialized(**inputs)
                second_boundary.verify_materialized(**inputs)
                first_memo = graph._VERIFICATION_OPERATION.get().entries[first_boundary]
                target = result.calls.count + 4

                def mutate(count):
                    if count == target:
                        first_memo.value[0].frame.person.loc[0, "unrelated_amount"] += 1

                result.calls.on_call = mutate
        assert first_boundary.revoked and second_boundary.revoked
    finally:
        result.calls.on_call = None


@pytest.mark.parametrize("bridge", ["spm", "handoff"])
def test_public_bridges_refuse_unissued_before_parent_traversal(tmp_path, bridge):
    from microcosm.build.us_runtime import current_survey_spm_source as spm
    from microcosm.build.us_runtime import native_survey_handoff as handoff

    destination = tmp_path / "not-written"
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        if bridge == "spm":
            spm.qualify_native_spm_inputs(
                object(), acs_profile=None, asec_scope_policy=None
            )
        else:
            handoff.write_native_survey_development_checkpoint(object(), destination)
    assert not destination.exists()


@pytest.mark.parametrize("phase", ["readback_owner", "closing_owner"])
def test_written_file_baseline_detects_real_child_owner_callback_mutation(
    actual, tmp_path, phase
):
    """Exact writer storage guard, not fabricated enrichment handoff authority."""
    from microcosm.build.us_runtime import native_survey_handoff as handoff

    result = graph_fixture._run(actual, tmp_path)
    destination = tmp_path / "checkpoint"
    frame = result.captured[graph.VERIFY].frame
    handoff._write_checkpoint(frame, {"invented_storage_only": True}, destination)
    expected = handoff._checkpoint_file_identity(destination)
    mutated = []
    try:
        with graph.verification_operation():
            result.check(result.warm, result.captured)
            loaded = handoff.load_native_survey_development_checkpoint(destination)
            assert loaded.owner_live_verified is False
            target = result.calls.count + 1

            def mutate(count):
                if count == target:
                    path = destination / "handoff.json"
                    # JSON remains semantically identical. Exact delivered
                    # bytes must nevertheless retain their pre-callback seal.
                    path.write_bytes(path.read_bytes() + b"\n")
                    mutated.append(phase)

            result.calls.on_call = mutate
            if phase == "readback_owner":
                result.check(result.warm, result.captured)
        assert mutated == [phase]
        handoff.same_replayed_frame(frame, loaded.frame)
        with pytest.raises(ValueError, match="NATIVE_HANDOFF_CHECKPOINT_CHANGED"):
            handoff._check_checkpoint_file_identity(destination, expected)
    finally:
        result.calls.on_call = None
