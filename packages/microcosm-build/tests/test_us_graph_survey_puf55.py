"""Actual invented originals, one extension, two routes and required replay.

The existing financial fixture is an actual issued postclone nineteen-node run. Original
PUF CSV construction is reused without running its CREATE fixture beforehand.
"""

import sys
from _thread import get_ident
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_puf55_canonical_donor import _original_sources
from test_us_puf55_survey_recipients import recipient_financial_run  # noqa: F401

from microcosm.build.us_runtime import graph_survey_puf55 as graph
from microcosm.graph import StructuralDelta
from microcosm.graph.keys import opaque_artifact_key
from microcosm.graph.store import StoreCorrupt


@pytest.fixture(scope="module")
def composed(recipient_financial_run, tmp_path_factory):  # noqa: F811
    upstream = recipient_financial_run
    root = tmp_path_factory.mktemp("survey_puf55_composition")
    definition, paths, _ = _original_sources(
        root / "original_puf", full_finalization_support=True
    )
    boundaries, finalizations, contexts = [], [], []

    # Code-local events preserve the exact captured objects without a Python
    # profile callback on every call/return throughout the 245-node graph.
    monitoring, tool_id = sys.monitoring, 4
    functions = (
        graph.attach.SurveyPuf55KeepAllKernel.run,
        graph.attach.SurveyPuf55MaskKernel.run,
        graph.attach.SurveyPuf55AttachKernel.run,
        graph.attach.numerical._finalize_puf55_routes,
        graph._construct,
    )
    codes = tuple(function.__code__ for function in functions)
    assert len({id(code) for code in codes}) == 5

    def started(code, instruction_offset):
        frame = sys._getframe(1)
        assert frame.f_code is code
        if code is codes[3]:
            finalizations.append(frame.f_locals["frame"])
        else:
            assert any(code is expected for expected in codes[:3])
            contexts.append(frame.f_locals["context"])

    def returned(code, instruction_offset, result):
        assert code is codes[4]
        if result is not None:
            boundaries.append(result)

    previous = sys.getprofile()
    # Refuse an occupied slot; never clear or borrow another tool's state.
    assert monitoring.get_tool(tool_id) is None
    monitoring.use_tool_id(tool_id, "microcosm-postclone-puf55-object-observer")
    try:
        assert monitoring.get_events(tool_id) == 0
        monitoring.register_callback(tool_id, monitoring.events.PY_START, started)
        monitoring.register_callback(tool_id, monitoring.events.PY_RETURN, returned)
        for code in codes[:4]:
            monitoring.set_local_events(tool_id, code, monitoring.events.PY_START)
        monitoring.set_local_events(tool_id, codes[4], monitoring.events.PY_RETURN)
        cold = graph.run_survey_puf55(
            upstream.run,
            donor_sources=paths,
            fixture_definition=definition,
            seed=578,
            n_estimators=2,
            zero_atol=0,
        )
        warm = graph.run_survey_puf55(
            upstream.run,
            donor_sources=paths,
            fixture_definition=definition,
            seed=578,
            n_estimators=2,
            zero_atol=0,
            resume="require",
        )
    finally:
        # Clear local events and callbacks even after setup/run failure, then
        # release only this fixture's claimed tool ID. No global events used.
        try:
            # CPython 3.14.4 retains local-event getter bits after clear alone.
            # Zero each of our exact local masks before clearing callbacks.
            for code in codes:
                monitoring.set_local_events(tool_id, code, 0)
            monitoring.clear_tool_id(tool_id)
            assert monitoring.get_events(tool_id) == 0
            assert all(
                monitoring.get_local_events(tool_id, code) == 0 for code in codes
            )
        finally:
            monitoring.free_tool_id(tool_id)
    assert monitoring.get_tool(tool_id) is None
    assert sys.getprofile() is previous
    assert all(
        function.__code__ is code
        for function, code in zip(functions, codes, strict=True)
    )
    assert len(boundaries) == len(finalizations) == 2
    assert len(contexts) == 3  # Required replay calls no attachment kernels.
    assert finalizations[0] is boundaries[0].expected.frame
    assert finalizations[1] is boundaries[1].expected.frame
    yield SimpleNamespace(
        upstream=upstream,
        cold=cold,
        warm=warm,
        boundary=boundaries[0],
        definition=definition,
        paths=paths,
        contexts={context.node.id: context for context in contexts},
    )
    for boundary in boundaries:
        boundary.pure()
    graph.financial.check_atomic_survey_financial_run(upstream.run)


@contextmanager
def _observe_exact_code_events(trace, *, starts=(), returns=()):
    """Test-only regular-Python events on this thread, preserving unwind returns.

    PY_UNWIND is global-only; ignore unlisted codes before acquiring frames.
    Existing profilers and other monitoring tools remain untouched.
    """
    monitoring, tool_id = sys.monitoring, 4
    functions = tuple(dict.fromkeys((*starts, *returns)))
    codes = tuple(function.__code__ for function in functions)
    start_codes = tuple(function.__code__ for function in starts)
    return_codes = tuple(function.__code__ for function in returns)
    assert functions and len({id(code) for code in codes}) == len(codes)
    # No generators/coroutines: their resume/yield events need another contract.
    assert not any(code.co_flags & (0x20 | 0x80 | 0x200) for code in codes)
    thread_id = get_ident()
    previous = sys.getprofile()

    def started(code, instruction_offset):
        if get_ident() != thread_id:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        assert any(code is selected for selected in start_codes)
        trace(frame, "call", None)

    def returned(code, instruction_offset, result):
        if get_ident() != thread_id:
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        assert any(code is selected for selected in return_codes)
        trace(frame, "return", result)

    def unwound(code, instruction_offset, exception):
        if get_ident() != thread_id or not any(
            code is selected for selected in return_codes
        ):
            return
        frame = sys._getframe(1)
        assert frame.f_code is code
        trace(frame, "return", None)

    assert monitoring.get_tool(tool_id) is None
    monitoring.use_tool_id(tool_id, "microcosm-puf55-negative-exact-events")
    try:
        assert monitoring.get_events(tool_id) == 0
        assert all(monitoring.get_local_events(tool_id, code) == 0 for code in codes)
        if start_codes:
            assert (
                monitoring.register_callback(
                    tool_id, monitoring.events.PY_START, started
                )
                is None
            )
        if return_codes:
            assert (
                monitoring.register_callback(
                    tool_id, monitoring.events.PY_RETURN, returned
                )
                is None
            )
            assert (
                monitoring.register_callback(
                    tool_id, monitoring.events.PY_UNWIND, unwound
                )
                is None
            )
        for code in codes:
            mask = 0
            if any(code is selected for selected in start_codes):
                mask |= monitoring.events.PY_START
            if any(code is selected for selected in return_codes):
                mask |= monitoring.events.PY_RETURN
            monitoring.set_local_events(tool_id, code, mask)
        if return_codes:
            monitoring.set_events(tool_id, monitoring.events.PY_UNWIND)
        yield
    finally:
        try:
            # Explicit zeroing matches the earlier 3.14.4 cleanup correction.
            monitoring.set_events(tool_id, 0)
            for code in codes:
                monitoring.set_local_events(tool_id, code, 0)
            monitoring.clear_tool_id(tool_id)
            assert monitoring.get_events(tool_id) == 0
            assert all(
                monitoring.get_local_events(tool_id, code) == 0 for code in codes
            )
        finally:
            monitoring.free_tool_id(tool_id)
        assert monitoring.get_tool(tool_id) is None
        assert sys.getprofile() is previous
        assert all(
            function.__code__ is code
            for function, code in zip(functions, codes, strict=True)
        )


def _values(case):
    b = case.boundary
    b.borrow()
    loaded = graph.financial._artifacts(
        case.cold.manifest,
        b.compiled,
        b.store,
        b.kernels,
        dict(b.keys),
        dict(b.implementations),
    )
    return graph._loaded_values(b, case.cold.manifest, loaded, b.nodes[2])


def test_actual_two_route_extension_and_required_replay(composed):
    case = composed
    cold, warm, run, b = case.cold, case.warm, case.upstream.run, case.boundary
    assert len(run.prefix.compiled.order) == 9
    assert len(run.compiled.order) == 19
    assert len(cold.compiled.order) == len(warm.compiled.order) == 245
    clone = graph.financial.atomic.clone
    expand = clone.COMBINED_CLONE_NODE
    claim = clone.COMBINED_CLONE_CLAIM_NODE
    assign = cold.compiled.graph.node("geography.assign")
    donor = cold.compiled.graph.node(graph.financial.financial.DONOR_NODE)
    edge = graph.financial.financial._geography_edge()
    order = cold.compiled.order
    assert cold.compiled.graph.node(expand).structural is StructuralDelta.EXPAND
    assert assign.population == expand
    assert claim in cold.compiled.predecessors[assign.id]
    assert (
        order.index(expand)
        < order.index(claim)
        < order.index(assign.id)
        < order.index("geography.derive")
        < order.index(edge.producer)
        < order.index(donor.id)
        < order.index(graph.financial.financial.ATTACH_NODE)
    )
    assert graph.codec.decode_json(assign.params["definition"].encode())[
        "identity"
    ] == [
        "survey_geography_origin_key",
        "household_support_clone_index",
    ]
    assert edge in donor.artifact_inputs
    assert edge.producer in cold.compiled.predecessors[donor.id]
    assert donor.base == graph.financial.survey.CREATE_NODE
    assert (
        run.manifest.population(donor.id).resolve_weights("person").kind
        is graph.financial.values.WeightKind.DESIGN
    )
    gate = cold.manifest.node(edge.producer)
    assert gate.opaque_artifacts[edge.artifact] == opaque_artifact_key(
        gate.key, edge.artifact
    )
    assert cold.store.load_bytes(gate.opaque_artifacts[edge.artifact]) == (
        graph.codec.encode_json(
            graph.codec.decode_json(run.projection)["atomic_geography"][
                "validation_receipt"
            ]
        )
    )
    assert run.prefix.clone_population is run.prefix.geography_population
    assert "census_block_geoid" not in run.prefix.expanded_population.frame.table(
        "household"
    )
    assert "census_block_geoid" not in run.prefix.allocated_population.frame.table(
        "household"
    )
    households = run.prefix.geography_population.frame.table("household")
    assert "census_block_geoid" in households
    pd.testing.assert_frame_equal(
        households,
        cold.population.frame.table("household")[list(households)],
        check_exact=True,
    )
    assert cold.manifest.key == warm.manifest.key
    assert tuple(r.profile for r in b.routes) == graph.attach.PROFILES
    assert tuple(name for name, _ in b.qualified.matrices) == tuple(
        p.value for p in graph.attach.PROFILES
    )
    decoded = [
        graph.recipient.model_input.decode_recipient_matrix(payload)
        for _, payload in b.qualified.matrices
    ]
    ids = [set(matrix.entity_ids.tolist()) for matrix in decoded]
    table = b.expected.frame.table("tax_unit")
    selected = graph.physical._masks(b.expected.frame)["tax_unit"]
    assert ids[0] and ids[1] and not ids[0] & ids[1]
    assert ids[0] | ids[1] == set(table.loc[selected, "tax_unit_id"])
    assert not (ids[0] | ids[1]) & set(table.loc[~selected, "tax_unit_id"])
    for matrix, route in zip(decoded, b.routes, strict=True):
        assert tuple(matrix.features.columns) == route.profile.predictors
        assert "employment_income_last_year" not in matrix.features
    assert all(cold.manifest.node(n).hit for n in run.compiled.order)
    assert all(record.hit for record in warm.manifest.nodes.values())
    assert len(cold.compiled.graph.sources) == 4
    assert cold.store.root == run.store.root and cold.store is not run.store
    assert cold.kernels is not run.kernels
    assert cold.store.codecs is not run.store.codecs
    for node, key in graph.financial._run_entry(run)[2].keys:
        assert cold.manifest.node(node).key == key
    assert cold.compiled.order.index(
        graph.financial.financial.ATTACH_NODE
    ) < cold.compiled.order.index(graph.attach.FILTER_NODE)
    assert all(
        fit.population == graph.canonical.CANONICAL_DONOR_NODE
        for route in b.routes
        for fit in route.fits
    )
    for route in b.routes:
        assert tuple(route.fits[0].inputs[0].columns) == (
            *route.profile.predictors,
            *route.profile.targets,
        )
        matrix = next(e for e in route.applies[0].artifact_inputs if e.name == "matrix")
        assert matrix.artifact == graph.recipient._NAMES[route.profile.value]
    donor = cold.manifest.population(graph.canonical.CANONICAL_DONOR_NODE)
    assert len(donor.table("tax_unit")) == 64
    assert donor.table("tax_unit").index.tolist() == list(range(501000, 501064))
    assert donor.table("tax_unit").tax_unit_id.tolist() == list(range(1, 65))
    assert donor.weights_for("tax_unit").values[0] == 0
    assert donor.weights_for("tax_unit").values[1:].tolist() == [
        (125 + i) / 100 for i in range(1, 64)
    ]
    graph.physical.replay.same_replayed_population(cold.population, warm.population)
    assert graph.codec.decode_json(cold.receipt)["release_eligible"] is False


def test_direct_whole_cohort_oracle_all_outputs_and_nonowned_storage(composed):
    case, b = composed, composed.boundary
    values = _values(case)
    _, donors, _ = b.canonical_donor(values)
    draws = tuple(b.histories(values))
    # Separate direct finalizer oracle; neither the attachment's selected-column
    # helper nor its retained computed KernelResult supplies expected values.
    raw, _ = graph.attach.numerical.merge_puf55_route_draws(
        b.expected.frame,
        recipient_matrices=b.qualified.matrices,
        route_draws=draws,
        seed=b.seed,
    )
    profile = graph.attach.PROFILES[0]
    expected = graph.attach.full.support.finalize_us_puf_tax_detail_predictions(
        b.expected.frame,
        donors[0],
        raw.copy(deep=True),
        person_outputs=profile.person_outputs,
        tax_unit_outputs=profile.tax_unit_outputs,
        tail_bound_diagnostics=[],
        absent_cells=graph.attach.full.support.PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    before, after = b.expected.frame, case.cold.population.frame
    masks = graph.physical._masks(before)
    # The survey's conserving interest split exists before PUF attachment.
    # PUF owns clone1 only; original-channel complements must survive exactly.
    interest = "tax_exempt_interest_income"
    assert interest in before.person
    assert before.person[interest].notna().all()
    pd.testing.assert_series_equal(
        before.person.loc[~masks["person"], interest],
        after.person.loc[~masks["person"], interest],
        check_exact=True,
    )
    coordinates = {(o.entity, o.column) for o in b.nodes[2].outputs}
    for entity in before.entities:
        left, right = before.table(entity), after.table(entity)
        assert left.index.identical(right.index)
        for column in left:
            owned = (entity, column) in coordinates
            unchanged = ~masks[entity] if owned else np.ones(len(left), dtype=bool)
            assert graph.population_ops.storage_equal(
                left[column], right[column], unchanged
            )
        for owned in b.nodes[2].outputs:
            if owned.entity != entity:
                continue
            ids = before.schema.entity_id_column(entity)
            actual = right.set_index(ids).loc[
                left.loc[masks[entity], ids], owned.column
            ]
            oracle = (
                expected.table(entity)
                .set_index(ids)
                .loc[actual.index, owned.column]
                .astype(owned.dtype)
            )
            pd.testing.assert_series_equal(actual, oracle, check_exact=True)
        np.testing.assert_array_equal(
            before.resolve_weights(entity).values, after.resolve_weights(entity).values
        )
    assert not any(
        o.column == "employment_income_last_year" for o in b.nodes[2].outputs
    )
    assert not set(graph.attach.full.SURVEY_SS_COMPONENTS) & {
        o.column for o in b.nodes[2].outputs
    }
    assert case.cold.population.mass_ledger == b.masked.mass_ledger
    assert (
        case.cold.population.design_weights.keys() == b.expected.design_weights.keys()
    )
    for entity in b.expected.design_weights:
        np.testing.assert_array_equal(
            case.cold.population.design_weights[entity],
            b.expected.design_weights[entity],
        )
    b.pure()


def test_wrong_actual_model_producer_is_refused_before_deserialization(composed):
    b, values = composed.boundary, _values(composed)
    name = "r0_model_000"
    value = values[name]
    wrong = "f" * 64 if value.producer_key != "f" * 64 else "e" * 64
    edge = next(e for e in b.nodes[2].artifact_inputs if e.name == name)
    values[name] = replace(
        value, producer_key=wrong, key=opaque_artifact_key(wrong, edge.artifact)
    )
    calls = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__.__code__
        ):
            calls.append(1)

    with _observe_exact_code_events(
        trace,
        starts=(
            graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__,
        ),
    ):
        with pytest.raises(ValueError, match="ARTIFACT_IDENTITY"):
            b.finalize(values)
    assert not calls
    b.pure()


@pytest.mark.parametrize(
    "mutation", ("receiving_after_finalizer", "selected_output_after_final_borrow")
)
def test_late_receiving_and_detached_output_mutation_refused(composed, mutation):
    b, values = composed.boundary, _values(composed)
    original_age = b.expected.frame.person.age.copy(deep=True)
    fired = []

    def trace(frame, event, arg):
        if fired or event != "return":
            return
        if (
            mutation == "receiving_after_finalizer"
            and frame.f_code is graph.attach.numerical._finalize_puf55_routes.__code__
        ):
            # Mutate the retained DataFrame, not an extracted Series that may
            # detach on write. Prove both the callee identity and changed owner.
            assert frame.f_locals["frame"] is b.expected.frame
            receiving = b.expected.frame.person
            receiving.iloc[0, receiving.columns.get_loc("age")] = (
                int(original_age.iloc[0]) + 1
            )
            assert b.expected.frame.person.age.iloc[0] != original_age.iloc[0]
            assert graph.physical._population_stamp(b.expected) != b.expected_stamp
            fired.append(1)
        elif (
            mutation == "selected_output_after_final_borrow"
            and frame.f_code is graph.attach.Boundary.borrow.__code__
        ):
            caller = frame.f_back
            if (
                caller is not None
                and caller.f_code is graph.attach.Boundary.finalize.__code__
                and "result" in caller.f_locals
            ):
                result = caller.f_locals["result"]
                coordinate = next(
                    k
                    for k, s in result.columns.items()
                    if s.dtype == np.dtype("float64")
                )
                changed = result.columns[coordinate].copy(deep=True)
                changed.iloc[0] += 123
                result.columns[coordinate] = changed
                fired.append(1)

    try:
        with _observe_exact_code_events(
            trace,
            returns=(
                (
                    graph.attach.numerical._finalize_puf55_routes
                    if mutation == "receiving_after_finalizer"
                    else graph.attach.Boundary.borrow
                ),
            ),
        ):
            with pytest.raises(ValueError, match="CHANGED"):
                b.finalize(values)
    finally:
        b.expected.frame.person["age"] = original_age
    assert fired == [1]
    b.pure()


def test_public_financial_dataclass_copy_cannot_issue_extension(composed):
    run = replace(composed.upstream.run)
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        graph._construct(
            run,
            composed.paths,
            fixture_definition=composed.definition,
            seed=578,
            n_estimators=2,
            zero_atol=0,
        )
    composed.boundary.pure()


@pytest.mark.parametrize("mutation", ("receipt", "artifact"))
def test_first_result_metadata_seal_binds_immutable_numerical_evidence(
    composed, mutation
):
    b = graph._construct(
        composed.upstream.run,
        composed.paths,
        fixture_definition=composed.definition,
        seed=578,
        n_estimators=2,
        zero_atol=0,
    )
    assert b.computed is None
    values = _values(composed)
    expected_stamp = graph.physical._population_stamp(b.expected)
    fired, finalizer_calls = [], []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is graph.attach.full.support.finalize_us_puf_tax_detail_predictions.__code__
        ):
            finalizer_calls.append(1)
        if (
            event == "return"
            and frame.f_code is graph.codec.encode_json.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is graph.attach.Boundary.finalize.__code__
            and "result" in frame.f_back.f_locals
            and not fired
        ):
            parent = frame.f_back.f_locals
            result = parent["result"]
            evidence = parent["evidence"]
            assert type(arg) is bytes and arg == evidence
            assert b.computed is None
            if mutation == "receipt":
                old = result.receipt["recipient_rows"]
                assert type(old) is int
                result.receipt["recipient_rows"] = old + 1
            else:
                assert result.artifacts == {"finalization": evidence}
                result.artifacts["finalization"] = b"changed-after-encoder-return"
            fired.append(1)
            assert graph.physical._population_stamp(b.expected) == expected_stamp

    with _observe_exact_code_events(
        trace,
        starts=(graph.attach.full.support.finalize_us_puf_tax_detail_predictions,),
        returns=(graph.codec.encode_json,),
    ):
        with pytest.raises(
            ValueError, match="^SURVEY_PUF55_NUMERICAL_RECEIPT_BINDING$"
        ):
            b.finalize(values)
    assert fired == finalizer_calls == [1]
    assert b.computed is None
    b.pure()
    composed.boundary.pure()


@pytest.mark.parametrize("mutation", ("sources", "declaration", "route_matrix"))
def test_actual_context_declarations_and_route_matrix_are_exact(composed, mutation):
    b = composed.boundary
    original = composed.contexts[graph.attach.ATTACH_NODE]
    if mutation == "sources":
        paths = dict(original.sources)
        name = next(iter(paths))
        paths[name] = next(p for p in paths.values() if p != paths[name])
        changed = replace(original, sources=paths)
        reason = "CONTEXT_SOURCE_PATHS"
    elif mutation == "declaration":
        node = replace(original.node, params={**original.node.params, "seed": 579})
        changed = replace(original, node=node, params=node.params)
        reason = "DECLARATION"
    else:
        artifacts = dict(original.artifacts)
        first, second = (graph.recipient._NAMES[p.value] for p in graph.attach.PROFILES)
        artifacts[first] = replace(artifacts[first], payload=artifacts[second].payload)
        changed = replace(original, artifacts=artifacts)
        reason = "RECIPIENT_MATRIX"
    with pytest.raises(ValueError, match=reason):
        b.context(changed, b.nodes[2], b.masked)
    b.context(original, b.nodes[2], b.masked)


@pytest.mark.parametrize("mutation", ("payload", "producer", "mutable_payload"))
def test_typed_artifact_mutation_during_source_borrow_refused_before_decode(
    composed, mutation
):
    b, values = composed.boundary, _values(composed)
    target = values["r0_model_000"]
    original = (target.payload, target.producer_key, target.key)
    fired, decodes = [], []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__.__code__
        ):
            decodes.append(1)
        if (
            event == "return"
            and frame.f_code is graph.attach.Boundary.canonical_donor.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is graph.attach.Boundary.finalize.__code__
            and not fired
        ):
            if mutation == "payload":
                object.__setattr__(target, "payload", b"changed typed model")
            elif mutation == "producer":
                wrong = "f" * 64 if target.producer_key != "f" * 64 else "e" * 64
                object.__setattr__(target, "producer_key", wrong)
                object.__setattr__(target, "key", opaque_artifact_key(wrong, "model"))
            else:
                object.__setattr__(target, "payload", bytearray(target.payload))
            fired.append(1)

    try:
        with _observe_exact_code_events(
            trace,
            starts=(
                graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__,
            ),
            returns=(graph.attach.Boundary.canonical_donor,),
        ):
            with pytest.raises(
                ValueError, match="ARTIFACT_(CHANGED_BEFORE_DECODE|VALUE_TYPE)"
            ):
                b.finalize(values)
    finally:
        for name, value in zip(
            ("payload", "producer_key", "key"), original, strict=True
        ):
            object.__setattr__(target, name, value)
    assert fired == [1]
    assert not decodes
    b.pure()


@pytest.mark.parametrize("mutation", ("capacity", "columns_metadata"))
def test_canonical_return_cannot_change_unmodeled_donor_surface(composed, mutation):
    b, values = composed.boundary, _values(composed)
    fired, decodes = [], []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__.__code__
        ):
            decodes.append(1)
        if (
            event == "return"
            and frame.f_code is graph.attach.Boundary.canonical_donor.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is graph.attach.Boundary.finalize.__code__
            and not fired
        ):
            _, donors, _ = arg
            for donor in donors:
                if mutation == "capacity":
                    column = "puf_person_incidence_capacity"
                    before = donor.iloc[0][column]
                    donor.iloc[0, donor.columns.get_loc(column)] = before + 1
                    assert donor.iloc[0][column] != before
                else:
                    donor.columns.name = "changed_after_source_reconstruction"
            fired.append(1)

    with _observe_exact_code_events(
        trace,
        starts=(
            graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__,
        ),
        returns=(graph.attach.Boundary.canonical_donor,),
    ):
        with pytest.raises(ValueError, match="DONOR_RESULT_CHANGED"):
            b.finalize(values)
    assert fired == [1]
    assert not decodes
    b.pure()


def test_matching_public_producer_labels_do_not_authorize_changed_bytes(composed):
    b, values = composed.boundary, _values(composed)
    name = "r0_model_000"
    values[name] = replace(values[name], payload=b"not the retained producer payload")
    decodes = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__.__code__
        ):
            decodes.append(1)

    with _observe_exact_code_events(
        trace,
        starts=(
            graph.physical.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__,
        ),
    ):
        with pytest.raises(ValueError, match="STORED_ARTIFACT_BYTES"):
            b.finalize(values)
    assert not decodes
    b.pure()


@pytest.mark.parametrize(
    "mutation",
    ("candidate_return", "decoded_receipt_and_candidate", "kernel_result_return"),
)
def test_first_numerical_return_cannot_change_selected_candidate(composed, mutation):
    # Construct a new actual-owner boundary over the already executed producers.
    # It has never accepted a numerical result: repeated-result comparison must
    # not mask this first-acceptance gap. No graph execution or fit is repeated.
    b = graph._construct(
        composed.upstream.run,
        composed.paths,
        fixture_definition=composed.definition,
        seed=578,
        n_estimators=2,
        zero_atol=0,
    )
    assert b.computed is None
    values = _values(composed)
    expected_stamp = graph.physical._population_stamp(b.expected)
    fired, finalizer_calls = [], []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is graph.attach.full.support.finalize_us_puf_tax_detail_predictions.__code__
        ):
            finalizer_calls.append(1)
        if (
            event == "return"
            and frame.f_back is not None
            and frame.f_back.f_code is graph.attach.Boundary.finalize.__code__
            and not fired
            and (
                (
                    mutation == "candidate_return"
                    and frame.f_code
                    is graph.attach.numerical._finalize_puf55_routes.__code__
                )
                or (
                    mutation == "decoded_receipt_and_candidate"
                    and frame.f_code is graph.codec.decode_json.__code__
                    and "candidate" in frame.f_back.f_locals
                )
                or (
                    mutation == "kernel_result_return"
                    and frame.f_code is graph.attach.KernelResult.__init__.__code__
                )
            )
        ):
            if mutation == "kernel_result_return":
                result = frame.f_locals["self"]
                series = next(
                    v for v in result.columns.values() if v.dtype == np.dtype("float64")
                )
                before = series.iloc[0]
                assert np.isfinite(before)
                series.iloc[0] = before + 123.0
                assert np.isfinite(series.iloc[0]) and series.iloc[0] != before
                assert graph.physical._population_stamp(b.expected) == expected_stamp
                fired.append(1)
                return
            if mutation == "candidate_return":
                candidate, receipt = arg
                assert type(receipt) is bytes
            else:
                candidate = frame.f_back.f_locals["candidate"]
                assert arg["protocol"] == "microcosm.us.puf55-route-finalization.v2"
            assert b.computed is None
            owned = next(o for o in b.nodes[2].outputs if o.dtype == "float64")
            mask = graph.physical._masks(b.expected.frame)[owned.entity]
            table = candidate.table(owned.entity)
            position = np.flatnonzero(mask)[0]
            column = table.columns.get_loc(owned.column)
            before = table.iloc[position, column]
            assert np.isfinite(before)
            table.iloc[position, column] = before + 123.0
            assert np.isfinite(table.iloc[position, column])
            assert table.iloc[position, column] != before
            assert graph.physical._population_stamp(b.expected) == expected_stamp
            if mutation == "decoded_receipt_and_candidate":
                arg["candidate_frame_sha256"] = (
                    graph.attach.numerical._candidate_frame_sha256(candidate)
                )
            fired.append(1)

    with _observe_exact_code_events(
        trace,
        starts=(graph.attach.full.support.finalize_us_puf_tax_detail_predictions,),
        returns=(
            {
                "candidate_return": graph.attach.numerical._finalize_puf55_routes,
                "decoded_receipt_and_candidate": graph.codec.decode_json,
                "kernel_result_return": graph.attach.KernelResult.__init__,
            }[mutation],
        ),
    ):
        code = {
            "candidate_return": "NUMERICAL_OUTPUT_CHANGED",
            "decoded_receipt_and_candidate": "NUMERICAL_RECEIPT_BINDING",
            "kernel_result_return": "NUMERICAL_COLUMN_BINDING",
        }[mutation]
        with pytest.raises(ValueError, match="^SURVEY_PUF55_" + code + "$"):
            b.finalize(values)
    assert fired == finalizer_calls == [1]
    assert b.computed is None
    b.pure()
    composed.boundary.pure()


@pytest.mark.parametrize(
    "constructor", [lambda: object(), lambda: graph.SurveyPuf55Run(*([None] * 8))]
)
def test_unissued_public_values_cannot_be_checked(constructor):
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        graph.check_survey_puf55_run(constructor())


def test_public_values_method_does_not_issue_authority():
    values = graph.SurveyPuf55Run(*([None] * 8))
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        values.checked_view()


def test_real_cold_and_required_handles_recheck_without_execution(composed):  # noqa: F811
    entered = []

    def trace(frame, event, arg):
        if event == "call":
            entered.append(frame.f_code.co_name)

    with _observe_exact_code_events(
        trace,
        starts=(
            graph.run_graph,
            graph.attach.Boundary.finalize,
            graph.attach.Boundary.canonical_donor,
        ),
    ):
        cold = composed.cold.checked_view()
        warm = graph.check_survey_puf55_run(composed.warm)
    assert entered == []
    assert cold.population is composed.cold.population
    assert warm.population is composed.warm.population
    assert cold.payload == warm.payload
    assert cold.digest == warm.digest == graph.codec.sha(cold.payload)
    document = graph.codec.decode_json(cold.payload)
    assert document["release_eligible"] is False
    assert document["population_admission_issued"] is False
    assert document["node_count"] == 245
    assert len(document["artifact_payload_sha256"]) > 220


@pytest.mark.parametrize("clone", [copy, replace])
def test_copied_real_run_is_unissued(composed, clone):  # noqa: F811
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        graph.check_survey_puf55_run(clone(composed.cold))
    graph._pure_run(composed.cold, graph._run_entry(composed.cold))


@pytest.mark.parametrize(
    "field",
    [
        "financial_run",
        "population",
        "manifest",
        "compiled",
        "store",
        "kernels",
        "sources",
        "receipt",
    ],
)
def test_retained_public_field_replacement_is_refused(composed, field):  # noqa: F811
    run = composed.cold
    original = getattr(run, field)
    entry = graph._run_entry(run)
    try:
        object.__setattr__(run, field, None)
        with pytest.raises(ValueError, match="PUF55_RUN_BINDINGS_CHANGED"):
            graph._pure_run(run, entry)
    finally:
        object.__setattr__(run, field, original)
    graph._pure_run(run, entry)


@pytest.mark.parametrize("surface", ["amount", "design_anchor", "owner", "mass_ledger"])
def test_retained_complete_population_mutation_is_refused(composed, surface):  # noqa: F811
    run, entry = composed.cold, graph._run_entry(composed.cold)
    population = run.population
    if surface == "amount":
        table = population.frame.person
        name = "age"
        before = table[name].copy(deep=True)
        table.loc[table.index[0], name] += 1

        def restore():
            table[name] = before

    elif surface == "design_anchor":
        before = population.design_weights
        changed = {name: vector.copy() for name, vector in before.items()}
        changed["household"][0] += 1
        object.__setattr__(population, "design_weights", changed)

        def restore():
            object.__setattr__(population, "design_weights", before)

    else:
        name = "owners" if surface == "owner" else "mass_ledger"
        before = getattr(population, name)
        assert before
        changed = dict(before) if surface == "owner" else ()
        if surface == "owner":
            changed[next(iter(changed))] = "invented_wrong_writer"
        object.__setattr__(population, name, changed)

        def restore():
            object.__setattr__(population, name, before)

    try:
        with pytest.raises(ValueError, match="PUF55_RUN_POPULATION_CHANGED"):
            graph._pure_run(run, entry)
    finally:
        restore()
    graph._pure_run(run, entry)


def test_compiled_execution_order_mutation_is_refused(composed):  # noqa: F811
    run, entry = composed.cold, graph._run_entry(composed.cold)
    original = run.compiled.order
    try:
        object.__setattr__(run.compiled, "order", original[::-1])
        with pytest.raises(ValueError, match="PUF55_RUN_DECLARATION_CHANGED"):
            graph._pure_run(run, entry)
    finally:
        object.__setattr__(run.compiled, "order", original)
    graph._pure_run(run, entry)


def test_attached_manifest_ledger_mutation_is_refused(composed):  # noqa: F811
    run, entry = composed.cold, graph._run_entry(composed.cold)
    original = run.manifest.mass_ledgers
    changed = dict(original)
    assert changed[run.population.version]
    changed[run.population.version] = ()
    try:
        object.__setattr__(run.manifest, "mass_ledgers", changed)
        with pytest.raises(ValueError, match="PUF55_RUN_MANIFEST_CHANGED"):
            graph._pure_run(run, entry)
    finally:
        object.__setattr__(run.manifest, "mass_ledgers", original)
    graph._pure_run(run, entry)


@pytest.mark.parametrize("field", ["populations", "mass_ledgers"])
def test_extra_attached_manifest_version_is_refused(composed, field):  # noqa: F811
    run, entry = composed.cold, graph._run_entry(composed.cold)
    original = getattr(run.manifest, field)
    changed = {**original, "invented_extra_version": next(iter(original.values()))}
    original_json = run.manifest.to_json()
    try:
        object.__setattr__(run.manifest, field, changed)
        assert run.manifest.to_json() == original_json
        with pytest.raises(ValueError, match="PUF55_RUN_MANIFEST_ROSTER_CHANGED"):
            graph._pure_run(run, entry)
    finally:
        object.__setattr__(run.manifest, field, original)
    graph._pure_run(run, entry)


def test_reused_artifact_verifier_refuses_actual_store_mutation(composed):  # noqa: F811
    run, boundary = composed.cold, composed.boundary
    key = run.manifest.node(graph.attach.ATTACH_NODE).opaque_artifacts["finalization"]
    path = run.store.object_path(key) / "payload.bin"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"invented corruption")
        with pytest.raises(StoreCorrupt):
            graph.financial._artifacts(
                run.manifest,
                run.compiled,
                run.store,
                run.kernels,
                dict(boundary.keys),
                dict(boundary.implementations),
            )
    finally:
        path.write_bytes(original)
    graph._pure_run(run, graph._run_entry(run))


def test_changed_original_donor_revokes_actual_run(composed):  # noqa: F811
    run = composed.cold
    path = next(iter(composed.paths.values()))
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError):
            graph.check_survey_puf55_run(run)
    finally:
        path.write_bytes(original)
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        graph.check_survey_puf55_run(run)


def test_late_io_output_mutation_revokes_actual_run(composed):  # noqa: F811
    run = composed.warm
    table = run.population.frame.person
    before = table.age.copy(deep=True)
    entry = graph._run_entry(run)
    mutated = False

    def trace(frame, event, arg):
        nonlocal mutated
        if (
            event == "return"
            and arg is not None
            and frame.f_locals.get("manifest") is run.manifest
        ):
            table.loc[table.index[0], "age"] += 1
            mutated = True

    try:
        with _observe_exact_code_events(trace, returns=(graph.financial._artifacts,)):
            with pytest.raises(ValueError, match="PUF55_RUN_POPULATION_CHANGED"):
                graph.check_survey_puf55_run(run)
    finally:
        table["age"] = before
    assert mutated
    assert graph._ISSUED_RUNS.get(id(run)) is not entry
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        graph.check_survey_puf55_run(run)
