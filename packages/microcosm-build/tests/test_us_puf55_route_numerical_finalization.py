"""Numerical bridge controls; invented local models, never graph/source authority.

The module fixture fits two small 55-target chains once through the maintained
QRF protocol. No issuer, graph, native source or trusted external model is mocked.
The original 18 raw-merger controls remain separate and byte-identical.
"""

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_puf55_route_finalization import _frame as raw_frame
from test_us_puf55_route_finalization import _inputs as raw_inputs
from test_us_puf55_survey_ss_profile import _donor55, _frame, _known55

from microcosm.build.us_runtime import puf55_route_finalization as bridge
from microcosm.fit import model_input, qrf, qrf_target

full = bridge.full
SEED = 21


def _donors():
    nine = _donor55()
    # This fixture's source IDs are deliberately shuffled RECIDs. The canonical
    # projection's existing pandas name is tax_unit_id, separate from model IDs.
    nine.index = pd.Index([11, 33, 22, 44], dtype="int64", name="tax_unit_id")
    return (nine, nine.drop(columns=full.SURVEY_SS_TOTAL_PREDICTOR))


def _fit(profile, donor):
    profile = full.require_puf_output_profile(profile)
    selected = full._validated_model_donor(donor, profile=profile)
    donor_frame = full.support._tax_unit_model_frame(selected)
    model = qrf.RegimeGatedQRF(seed=SEED, n_estimators=2, zero_atol=0)
    initial = model.start_chain(
        donor_frame, list(profile.predictors), list(profile.targets), weights="design"
    )
    state = qrf_target.LegacyQRFTrainingState.from_chain(initial)
    artifacts, rows = [], []
    for target in profile.targets:
        fitted = qrf_target.fit_target(
            model, donor_frame, state=state, weights="design"
        )
        assert fitted.target == target
        payload = fitted.to_bytes()
        artifacts.append((fitted, payload))
        rows.append(
            {
                "target": target,
                "sha256": full.codec.sha(payload),
                "training_id": fitted.training_id,
            }
        )
        state = fitted.next_training_state
    training = full.codec.encode_json(
        {"schema_version": 1, "state": state.to_dict(), "models": rows}
    )
    return SimpleNamespace(
        profile=profile,
        donor=donor,
        initial=initial,
        models=tuple(artifacts),
        model_rows=rows,
        training=training,
    )


def _apply(fitted, ids):
    profile = fitted.profile
    index = pd.Index(np.asarray(ids, dtype="int64"), name="tax_unit_id")
    features = pd.DataFrame(1.0, index=index, columns=profile.predictors)
    matrix = model_input.encode_recipient_matrix(
        features, entity="tax_unit", entity_ids=index.to_numpy()
    )
    producer = full.codec.sha(("invented route matrix:" + profile.value).encode())
    state = fitted.initial
    prior = pd.DataFrame(index=index)
    raw = []
    for artifact, _ in fitted.models:
        step = qrf_target.apply_target(artifact, features, prior, state=state)
        prior[step.target] = step.raw_draw
        raw.append(
            (
                step.target,
                full.codec.encode_raw_target(
                    step.raw_draw, target=step.target, index=index
                ),
            )
        )
        state = step.state
    application = {
        "schema_version": 1,
        "state": state.to_dict(),
        "seed": SEED,
        "models": fitted.model_rows,
        "raw_targets": [
            {"target": target, "sha256": full.codec.sha(payload)}
            for target, payload in raw
        ],
    }
    applied = full.codec.encode_json(
        {
            "schema_version": 1,
            "matrix_sha256": full.codec.sha(matrix),
            "matrix_producer_key": producer,
            "application": application,
        }
    )
    draws = bridge.Puf55RouteDraws(
        profile, matrix, producer, tuple(raw), applied, fitted.training
    )
    return bridge.Puf55RouteFinalizationInput(
        draws, fitted.donor, fitted.models[-1][1], profile.phase
    )


@pytest.fixture(scope="module")
def numerical_routes():
    frame = _frame()
    frame.person["invented_survey_ss_known"] = pd.array(
        [True, False, None, True] * 4, dtype="boolean"
    )
    mask = full.support.puf_tax_detail_clone_mask(
        frame.table("tax_unit"), entity="tax_unit"
    )
    ids = frame.table("tax_unit").loc[mask, "tax_unit_id"].to_numpy()
    # The eight route intentionally has no usable tax-unit SS total. A hidden
    # attempt to prepare one complete nine-feature matrix would therefore fail.
    frame.table("tax_unit").loc[
        frame.table("tax_unit").tax_unit_id.isin(ids[[3, 1]]),
        full.SURVEY_SS_TOTAL_PREDICTOR,
    ] = np.nan
    fitted = tuple(
        _fit(profile, donor)
        for profile, donor in zip(bridge.PROFILES, _donors(), strict=True)
    )
    # Each route is a genuine subset matrix. No complete nine-feature recipient
    # matrix is synthesized and no clone membership is changed for a route.
    packets = (_apply(fitted[0], ids[[2, 0]]), _apply(fitted[1], ids[[3, 1]]))
    return SimpleNamespace(frame=frame, ids=ids, fitted=fitted, routes=packets)


def _call(frame, items, **changes):
    kwargs = {
        "recipient_matrices": tuple(
            (item.draws.profile.value, item.draws.matrix) for item in items
        ),
        "routes": items,
        "seed": SEED,
        **changes,
    }
    return bridge._finalize_puf55_routes(frame, **kwargs)


def _assert_preserved(before, after):
    assert before.schema == after.schema and before.metadata == after.metadata
    assert before.mass_log == after.mass_log and before.links == after.links
    pd.testing.assert_series_equal(before.strata, after.strata, check_exact=True)
    for entity in before.weighted_entities:
        assert before.weights_for(entity).kind == after.weights_for(entity).kind
        np.testing.assert_array_equal(
            before.weights_for(entity).values, after.weights_for(entity).values
        )
    for entity in before.entities:
        owned = (
            full.PUF55_SURVEY_SS.person_outputs
            if entity == "person"
            else full.PUF55_SURVEY_SS.tax_unit_outputs
            if entity == "tax_unit"
            else ()
        )
        columns = [column for column in before.table(entity) if column not in owned]
        pd.testing.assert_frame_equal(
            before.table(entity)[columns],
            after.table(entity)[columns],
            check_exact=True,
        )
    for name in full.SURVEY_SS_COMPONENTS:
        assert (
            before.person[name].to_numpy().tobytes()
            == after.person[name].to_numpy().tobytes()
        )
    pd.testing.assert_series_equal(
        before.person.invented_survey_ss_known,
        after.person.invented_survey_ss_known,
        check_exact=True,
    )


def test_two_real_chains_finalize_once_on_the_complete_original_cohort(
    numerical_routes,
):
    case = numerical_routes
    calls, old = [], sys.getprofile()

    def observe(call, event, arg):
        if (
            event == "call"
            and call.f_code
            is full.support.finalize_us_puf_tax_detail_predictions.__code__
        ):
            calls.append(call.f_locals["predictions"].index.copy())

    try:
        sys.setprofile(observe)
        candidate, payload = _call(case.frame, case.routes)
    finally:
        sys.setprofile(old)
    raw, _ = bridge.merge_puf55_route_draws(
        case.frame,
        recipient_matrices=tuple(
            (r.draws.profile.value, r.draws.matrix) for r in case.routes
        ),
        route_draws=tuple(r.draws for r in case.routes),
        seed=SEED,
    )
    assert len(calls) == 1 and calls[0].identical(raw.index)
    expected = full.support.finalize_us_puf_tax_detail_predictions(
        case.frame,
        full._validated_model_donor(case.routes[0].donor, profile=bridge.PROFILES[0]),
        raw,
        person_outputs=bridge.PROFILES[0].person_outputs,
        tax_unit_outputs=bridge.PROFILES[0].tax_unit_outputs,
        tail_bound_diagnostics=[],
        absent_cells=full.support.PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    for entity in candidate.entities:
        pd.testing.assert_frame_equal(
            candidate.table(entity), expected.table(entity), check_exact=True
        )
    _assert_preserved(case.frame, candidate)
    receipt = full.codec.decode_json(payload)
    assert receipt["protocol"] == "microcosm.us.puf55-route-finalization.v2"
    assert receipt["candidate_frame_sha256"] == bridge._candidate_frame_sha256(
        candidate
    )
    assert receipt["recipient_rows"] == 4 and receipt["finalizer_calls"] == 1
    assert receipt["model_consumed_donor_values_checked"] is True
    assert [row["profile"] for row in receipt["routes"]] == [
        p.value for p in bridge.PROFILES
    ]
    assert len({row["common_donor_numerical_sha256"] for row in receipt["routes"]}) == 1
    assert not any(
        receipt[key]
        for key in (
            "source_admission_issued",
            "population_admission_issued",
            "release_eligible",
        )
    )


@pytest.mark.parametrize("route_index", (0, 1))
def test_one_nonempty_route_uses_same_whole_cohort_finalizer(
    numerical_routes, route_index
):
    case = numerical_routes
    # A separate ordinary source-value fixture represents this all-nine or
    # all-eight universe; never alter/trim the two-route receiving cohort.
    frame = _frame()
    frame.person["invented_survey_ss_known"] = pd.array(
        [True, False, None, True] * 4, dtype="boolean"
    )
    if route_index == 1:
        frame.table("tax_unit")[full.SURVEY_SS_TOTAL_PREDICTOR] = np.nan
    route = _apply(case.fitted[route_index], case.ids[::-1])
    candidate, payload = _call(frame, (route,))
    receipt = full.codec.decode_json(payload)
    assert receipt["protocol"] == "microcosm.us.puf55-route-finalization.v2"
    assert receipt["candidate_frame_sha256"] == bridge._candidate_frame_sha256(
        candidate
    )
    assert receipt["recipient_rows"] == 4 and len(receipt["routes"]) == 1
    assert receipt["routes"][0]["profile"] == bridge.PROFILES[route_index].value
    raw, _ = bridge.merge_puf55_route_draws(
        frame,
        recipient_matrices=((route.draws.profile.value, route.draws.matrix),),
        route_draws=(route.draws,),
        seed=SEED,
    )
    expected = full.support.finalize_us_puf_tax_detail_predictions(
        frame,
        full._validated_model_donor(route.donor, profile=route.draws.profile),
        raw,
        person_outputs=route.draws.profile.person_outputs,
        tax_unit_outputs=route.draws.profile.tax_unit_outputs,
        tail_bound_diagnostics=[],
        absent_cells=full.support.PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    for entity in candidate.entities:
        pd.testing.assert_frame_equal(
            candidate.table(entity), expected.table(entity), check_exact=True
        )
    _assert_preserved(frame, candidate)


@pytest.mark.parametrize(
    "defect",
    ("recid_order", "target", "weight", "capacity", "shared_money", "shared_status"),
)
def test_donor_parity_refuses_before_any_model_decode(defect):
    donors, draws = _donors(), raw_inputs()
    changed = donors[1].copy(deep=True)
    if defect == "recid_order":
        changed = changed.iloc[[1, 0, 2, 3]].copy()
    else:
        column = {
            "target": bridge.PROFILES[1].targets[-1],
            "weight": "weight",
            "capacity": "puf_person_incidence_capacity",
            "shared_money": bridge.PROFILES[1].predictors[2],
            "shared_status": bridge.PROFILES[1].predictors[0],
        }[defect]
        changed.iloc[0, changed.columns.get_loc(column)] += 1
    items = tuple(
        bridge.Puf55RouteFinalizationInput(
            draw, donor, b"must not be deserialized", draw.profile.phase
        )
        for draw, donor in zip(draws, (donors[0], changed), strict=True)
    )
    old, decoded = sys.getprofile(), False

    def observe(call, event, arg):
        nonlocal decoded
        if (
            event == "call"
            and call.f_code
            is qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes.__func__.__code__
        ):
            decoded = True

    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError, match="PUF55_ROUTE_DONOR_PARITY"):
            _call(raw_frame(), items)
    finally:
        sys.setprofile(old)
    assert not decoded


@pytest.mark.parametrize(
    "defect",
    (
        "phase",
        "model_seed",
        "matrix",
        "producer",
        "raw_history",
        "last_model",
        "ninth_donor_value",
        "both_donor_targets",
        "both_recid_order",
        "both_weights",
    ),
)
def test_route_state_and_actual_model_consumption_are_bound(numerical_routes, defect):
    case = numerical_routes
    first, second = case.routes
    expected = tuple((r.draws.profile.value, r.draws.matrix) for r in case.routes)
    if defect == "phase":
        first = replace(first, phase=second.phase)
    elif defect == "model_seed":
        training = full.codec.decode_json(first.draws.training_state)
        training["state"]["model_config"]["seed"] = SEED + 1
        first = replace(
            first,
            draws=replace(first.draws, training_state=full.codec.encode_json(training)),
        )
    elif defect == "matrix":
        expected = ((first.draws.profile.value, second.draws.matrix), expected[1])
    elif defect == "producer":
        first = replace(first, draws=replace(first.draws, matrix_producer_key="f" * 64))
    elif defect == "raw_history":
        target, payload = first.draws.raw_draws[0]
        matrix = model_input.decode_recipient_matrix(first.draws.matrix)
        values = full.codec.read_raw_target(
            payload, target=target, index=matrix.features.index
        ).copy()
        values[0] += 1
        changed = full.codec.encode_raw_target(
            values, target=target, index=matrix.features.index
        )
        first = replace(
            first,
            draws=replace(
                first.draws, raw_draws=((target, changed), *first.draws.raw_draws[1:])
            ),
        )
    elif defect == "last_model":
        first = replace(first, last_model=second.last_model)
    elif defect == "ninth_donor_value":
        donor = first.donor.copy(deep=True)
        donor.loc[11, full.SURVEY_SS_TOTAL_PREDICTOR] += 1
        first = replace(first, donor=donor)
    elif defect in ("both_recid_order", "both_weights"):
        donors = [item.donor.copy(deep=True) for item in (first, second)]
        if defect == "both_recid_order":
            donors = [donor.iloc[[1, 0, 2, 3]].copy() for donor in donors]
        else:
            for donor in donors:
                donor.loc[11, "weight"] += 1
        first, second = (
            replace(first, donor=donors[0]),
            replace(second, donor=donors[1]),
        )
    else:
        donors = [item.donor.copy(deep=True) for item in (first, second)]
        for donor in donors:
            donor.loc[11, bridge.PROFILES[0].targets[-1]] += 1
        first, second = (
            replace(first, donor=donors[0]),
            replace(second, donor=donors[1]),
        )
    expected_error = {
        "both_recid_order": "QRF chain donor index/order changed since start_chain",
        "both_weights": "QRF chain resolved weight values/order changed",
    }.get(
        defect,
        "PUF55_ROUTE_(PHASE|MODEL_SEED|RECIPIENT_MATRIX_CHANGED)|PUF_FULL_MATRIX_BINDING|PUF_RAW_HISTORY|Legacy QRF artifact content digest mismatch|PUF_DONOR_CONSUMED_BYTES",
    )
    with pytest.raises(ValueError, match=expected_error):
        _call(case.frame, (first, second), recipient_matrices=expected)


@pytest.mark.parametrize("defect", ("no_routes", "no_clone_one"))
def test_empty_cohort_and_empty_route_roster_refuse_before_models(defect):
    frame = raw_frame()
    items = ()
    if defect == "no_clone_one":
        column = bridge.provenance.support_clone_index_column("tax_unit")
        frame.table("tax_unit")[column] = np.zeros(
            len(frame.table("tax_unit")), dtype="int64"
        )
        draw = raw_inputs()[0]
        items = (
            bridge.Puf55RouteFinalizationInput(
                draw, _donors()[0], b"not trusted", draw.profile.phase
            ),
        )
    with pytest.raises(
        ValueError, match="PUF55_ROUTE_(FINALIZATION_ROUTES|NO_RECIPIENTS)"
    ):
        _call(frame, items)


@pytest.mark.parametrize("field", ("target", "capacity"))
def test_last_model_return_cannot_change_a_retained_donor(numerical_routes, field):
    case = numerical_routes
    items = tuple(
        replace(item, donor=item.donor.copy(deep=True)) for item in case.routes
    )
    old, fired = sys.getprofile(), False

    def observe(call, event, arg):
        nonlocal fired
        if event == "return" and call.f_code is full._check_fitted_model_donor.__code__:
            caller = call.f_back
            if (
                caller is not None
                and caller.f_code is bridge._finalize_puf55_routes.__code__
            ):
                fired = True
                column = (
                    bridge.PROFILES[0].targets[-1]
                    if field == "target"
                    else "puf_person_incidence_capacity"
                )
                items[0].donor.loc[11, column] += 1

    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError, match="PUF55_ROUTE_FINALIZATION_INPUT_CHANGED"):
            _call(case.frame, items)
    finally:
        sys.setprofile(old)
    assert fired


def test_model_donor_frame_matches_existing_single_profile_preparation():
    frame, donor = _frame(), _donors()[0]
    prepared = full.prepare_full_puf_inputs(
        frame, donor, predictor_known=_known55(frame), profile=full.PUF55_SURVEY_SS
    )
    actual = bridge._model_donor_frame(prepared.donor)
    expected = prepared.donor_frame
    assert actual.schema == expected.schema
    assert actual.metadata == expected.metadata and actual.mass_log == expected.mass_log
    assert actual.links == expected.links
    pd.testing.assert_series_equal(actual.strata, expected.strata, check_exact=True)
    for entity in actual.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity), expected.table(entity), check_exact=True
        )
    assert actual.weights_for("tax_unit").kind == expected.weights_for("tax_unit").kind
    np.testing.assert_array_equal(
        actual.weights_for("tax_unit").values, expected.weights_for("tax_unit").values
    )
    assert actual.table("tax_unit").index.tolist() == [11, 33, 22, 44]
    assert actual.weights_for("tax_unit").values.tolist() == [1.0, 0.0, 2.0, 3.0]


def test_model_frame_constructor_return_cannot_coerce_a_donor_column():
    selected = full._validated_model_donor(_donors()[0], profile=full.PUF55_SURVEY_SS)
    old, fired = sys.getprofile(), False

    def observe(call, event, arg):
        nonlocal fired
        if (
            event == "return"
            and call.f_code is full.support._tax_unit_model_frame.__code__
        ):
            if (
                call.f_back is not None
                and call.f_back.f_code is bridge._model_donor_frame.__code__
            ):
                fired = True
                column = full.PUF55_SURVEY_SS.predictors[0]
                arg.table("tax_unit")[column] = arg.table("tax_unit")[column].astype(
                    str
                )

    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError, match="PUF55_ROUTE_MODEL_DONOR_CONVERSION"):
            bridge._model_donor_frame(selected)
    finally:
        sys.setprofile(old)
    assert fired


@pytest.mark.parametrize("target", ("raw", "donor"))
def test_finalizer_return_cannot_change_retained_raw_or_donor(numerical_routes, target):
    case = numerical_routes
    items = tuple(
        replace(item, donor=item.donor.copy(deep=True)) for item in case.routes
    )
    old, fired = sys.getprofile(), False

    def observe(call, event, arg):
        nonlocal fired
        if (
            event == "return"
            and call.f_code
            is full.support.finalize_us_puf_tax_detail_predictions.__code__
            and call.f_back is not None
            and call.f_back.f_code is bridge._finalize_puf55_routes.__code__
            and not fired
        ):
            fired = True
            if target == "raw":
                call.f_back.f_locals["combined"].iloc[0, 0] += 1
            else:
                items[0].donor.loc[11, bridge.PROFILES[0].targets[-1]] += 1

    try:
        sys.setprofile(observe)
        with pytest.raises(
            ValueError,
            match=(
                "PUF55_ROUTE_FINALIZATION_RAW_CHANGED"
                if target == "raw"
                else "PUF55_ROUTE_FINALIZATION_INPUT_CHANGED"
            ),
        ):
            _call(case.frame, items)
    finally:
        sys.setprofile(old)
    assert fired


def test_receipt_encoding_cannot_change_the_already_sealed_candidate(numerical_routes):
    case = numerical_routes
    fired, calls = [], []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is full.support.finalize_us_puf_tax_detail_predictions.__code__
        ):
            calls.append(1)
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code is full.codec.encode_json.__code__
            and caller is not None
            and caller.f_code is bridge._finalize_puf55_routes.__code__
            and "candidate_sha256" in caller.f_locals
            and not fired
        ):
            candidate = caller.f_locals["candidate"]
            table = candidate.person
            mask = full.support.puf_tax_detail_clone_mask(table, entity="person")
            row = np.flatnonzero(mask)[0]
            target = "employment_income_before_lsr"
            assert target in bridge.PROFILES[0].person_outputs
            assert table[target].dtype == np.dtype("float64")
            column = table.columns.get_loc(target)
            before = table.iloc[row, column]
            assert np.isfinite(before)
            table.iloc[row, column] = before + 123.0
            assert (
                np.isfinite(table.iloc[row, column])
                and table.iloc[row, column] != before
            )
            fired.append(1)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(
            ValueError, match="^PUF55_ROUTE_FINALIZATION_CANDIDATE_CHANGED$"
        ):
            _call(case.frame, case.routes)
    finally:
        sys.setprofile(previous)
    assert fired == calls == [1]
