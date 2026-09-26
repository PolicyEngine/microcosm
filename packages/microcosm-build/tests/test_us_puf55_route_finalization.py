"""Invented real codec packets; no fitted model, source or graph authority.

These controls exercise only the route-value boundary. Actual train/apply,
donor/producer ownership and full attachment require separate guarded tests.
"""

import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf55_route_finalization as routes
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

full = routes.full
SEED = 21


def _frame(reverse=False):
    ids = np.array([11, 12, 13, 101, 202, 303], dtype=np.int64)
    person = pd.DataFrame({"person_id": ids})
    for entity in US_SCHEMA.group_entities:
        person[US_SCHEMA.membership_column(entity)] = ids
    person["social_security_source_value"] = [-0.0, np.nan, 17.0, np.nan, 0.0, 2.0]
    person["source_component_allowed"] = [True, False, True, False, False, True]
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({entity + "_id": ids})
    tax_unit = tables["tax_unit"]
    tax_unit[routes.provenance.support_clone_index_column("tax_unit")] = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int64
    )
    # Group entity IDs stay sorted, as required by Frame linkage validation.
    # Nonmonotonic pandas labels exercise the distinct receiving row axis.
    labels = [10, 20, 50, 70, 30, 90]
    tax_unit.index = pd.Index(labels[::-1] if reverse else labels, name="original_row")
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
        metadata={"source_admission": False, "scope": "invented_codec_values"},
    )


def _route(profile, ids, offset):
    """Build canonical synthetic chain metadata with the maintained codecs."""
    index = pd.Index(np.asarray(ids, dtype=np.int64), name="tax_unit_id")
    features = pd.DataFrame(1.0, index=index, columns=profile.predictors)
    matrix = model_input.encode_recipient_matrix(
        features, entity="tax_unit", entity_ids=index.to_numpy()
    )
    key = full.codec.sha(("invented-matrix:" + profile.value).encode())
    raw = tuple(
        (
            target,
            full.codec.encode_raw_target(
                np.asarray(ids, dtype=np.float64) + offset + j / 100,
                target=target,
                index=index,
            ),
        )
        for j, target in enumerate(profile.targets)
    )
    state = qrf.QRFChainState(
        predictors=profile.predictors,
        targets=profile.targets,
        completed_targets=profile.targets,
        entity="tax_unit",
        weight_kind="design",
        weight_sha256="1" * 64,
        n_estimators=2,
        zero_atol=1e-8,
        max_samples_leaf=None,
        max_samples_leaf_kind="none",
        seed=SEED,
        fit_n_jobs=1,
        donor_index=qrf._index_identity(pd.Index([1, 2], name="RECID")),
        recipient_index=qrf._index_identity(index),
        fit_rng_state_json=json.dumps(np.random.PCG64(1).state),
        draw_rng_state_json=json.dumps(np.random.PCG64(2).state),
    )
    models = [
        {
            "target": target,
            "sha256": full.codec.sha((profile.value + target).encode()),
            "training_id": full.codec.sha(
                ("training:" + profile.value + target).encode()
            ),
        }
        for target in profile.targets
    ]
    training = full.codec.encode_json(
        {
            "schema_version": 1,
            "state": qrf_target.LegacyQRFTrainingState.from_chain(state).to_dict(),
            "models": models,
        }
    )
    application = {
        "schema_version": 1,
        "state": state.to_dict(),
        "seed": SEED,
        "models": models,
        "raw_targets": [
            {"target": target, "sha256": full.codec.sha(payload)}
            for target, payload in raw
        ],
    }
    applied = full.codec.encode_json(
        {
            "schema_version": 1,
            "matrix_sha256": full.codec.sha(matrix),
            "matrix_producer_key": key,
            "application": application,
        }
    )
    return routes.Puf55RouteDraws(profile, matrix, key, raw, applied, training)


def _inputs():
    return (
        _route(full.PUF55_SURVEY_SS, [202, 303], 1000),
        _route(full.PUF55_SURVEY_SS_NO_TOTAL, [101], 2000),
    )


def _merge(frame, packets, **kwargs):
    arguments = {
        "recipient_matrices": tuple((p.profile.value, p.matrix) for p in packets),
        "route_draws": packets,
        "seed": SEED,
        **kwargs,
    }
    return routes.merge_puf55_route_draws(frame, **arguments)


@pytest.mark.parametrize("reverse", (False, True))
def test_merge_restores_complete_receiving_order_and_preserves_source_values(reverse):
    frame, packets = _frame(reverse), _inputs()
    before = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    person_bits = frame.person.social_security_source_value.to_numpy().tobytes()
    actual, payload = _merge(frame, packets)
    units = frame.table("tax_unit")
    selected = units.loc[units.tax_unit_id.isin([101, 202, 303])]
    expected = pd.DataFrame(
        {
            target: [
                value + (2000 if value == 101 else 1000) + j / 100
                for value in selected.tax_unit_id
            ]
            for j, target in enumerate(full.PUF55_SURVEY_SS.targets)
        },
        index=selected.index,
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.shape == (3, 55)
    receipt = full.codec.decode_json(payload)
    assert receipt["recipient_rows"] == 3
    assert [r["rows"] for r in receipt["routes"]] == [2, 1]
    assert not any(
        receipt[name]
        for name in (
            "finalization_performed",
            "donor_model_binding_verified_here",
            "source_admission_issued",
            "population_admission_issued",
            "release_eligible",
        )
    )
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            frame.table(entity), before[entity], check_exact=True
        )
    assert frame.person.social_security_source_value.to_numpy().tobytes() == person_bits


@pytest.mark.parametrize("profile", routes.PROFILES)
def test_single_nonempty_route_omits_other_route(profile):
    packet = _route(profile, [202, 101, 303], 1000)
    actual, payload = _merge(_frame(), (packet,))
    assert actual.index.tolist() == [70, 30, 90]
    assert len(full.codec.decode_json(payload)["routes"]) == 1


@pytest.mark.parametrize("defect", ("overlap", "missing", "extra", "clone_zero"))
def test_refuses_inexact_recipient_partition(defect):
    nine_ids, eight_ids = {
        "overlap": ([202, 303], [101, 202]),
        "missing": ([202], [101]),
        "extra": ([202, 303], [101, 999]),
        "clone_zero": ([202, 303], [101, 11]),
    }[defect]
    packets = (
        _route(full.PUF55_SURVEY_SS, nine_ids, 1000),
        _route(full.PUF55_SURVEY_SS_NO_TOTAL, eight_ids, 2000),
    )
    with pytest.raises(
        ValueError, match="PUF55_ROUTE_(OVERLAPPING_IDS|INCOMPLETE_RECIPIENT_UNION)"
    ):
        _merge(_frame(), packets)


@pytest.mark.parametrize("defect", ("matrix", "order", "mutable_bytes", "profile"))
def test_refuses_changed_matrix_route_or_storage(defect):
    packets = _inputs()
    expected = tuple((p.profile.value, p.matrix) for p in packets)
    if defect == "matrix":
        expected = (
            (expected[0][0], _route(packets[0].profile, [303, 202], 1000).matrix),
            expected[1],
        )
    elif defect == "order":
        packets = packets[::-1]
        expected = expected[::-1]
    elif defect == "mutable_bytes":
        packets = (replace(packets[0], matrix=bytearray(packets[0].matrix)), packets[1])
    else:
        packets = (replace(packets[0], profile=full.PUF59), packets[1])
    with pytest.raises(ValueError, match="PUF55_ROUTE_"):
        _merge(_frame(), packets, recipient_matrices=expected)


@pytest.mark.parametrize("defect", ("seed", "raw", "model_history", "target_history"))
def test_complete_chain_history_is_checked_before_merge(defect):
    packets = _inputs()
    if defect == "seed":
        with pytest.raises(ValueError, match="PUF_FULL_CHAIN_IDENTITY"):
            _merge(_frame(), packets, seed=SEED + 1)
        return
    packet = packets[0]
    if defect == "raw":
        target, _ = packet.raw_draws[0]
        changed = full.codec.encode_raw_target(
            [99.0, 98.0],
            target=target,
            index=model_input.decode_recipient_matrix(packet.matrix).features.index,
        )
        packet = replace(packet, raw_draws=((target, changed), *packet.raw_draws[1:]))
    else:
        applied = full.codec.decode_json(packet.apply_state)
        if defect == "model_history":
            applied["application"]["models"][0]["sha256"] = "e" * 64
        else:
            applied["application"]["raw_targets"].reverse()
        packet = replace(packet, apply_state=full.codec.encode_json(applied))
    with pytest.raises(
        ValueError, match="PUF_RAW_HISTORY|PUF_FULL_CHAIN_IDENTITY|Legacy QRF history"
    ):
        _merge(_frame(), (packet, packets[1]))


def test_changed_detached_decoder_values_refuse():
    old, fired = sys.getprofile(), False

    def observe(call, event, arg):
        nonlocal fired
        if event == "return" and call.f_code is full.decode_full_puf_draws.__code__:
            fired = True
            arg.iloc[0, 0] += 1

    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError, match="PUF55_ROUTE_DECODED_RAW_CHANGED"):
            _merge(_frame(), _inputs())
    finally:
        sys.setprofile(old)
    assert fired


def test_later_decoder_cannot_change_an_earlier_route_table():
    old, calls, fired = sys.getprofile(), 0, False

    def observe(call, event, arg):
        nonlocal calls, fired
        if event == "return" and call.f_code is full.decode_full_puf_draws.__code__:
            calls += 1
            if calls == 2:
                call.f_back.f_locals["tables"][0].iloc[0, 0] += 1
                fired = True

    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError, match="PUF55_ROUTE_MERGED_RAW_CHANGED"):
            _merge(_frame(), _inputs())
    finally:
        sys.setprofile(old)
    assert calls == 2 and fired
