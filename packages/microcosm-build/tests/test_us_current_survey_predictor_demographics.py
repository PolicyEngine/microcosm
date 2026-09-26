"""Opted-in financial conditioning over actual issuers and invented sources."""

import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments

from microcosm.build.us_runtime import current_survey_predictors as values
from microcosm.build.us_runtime import graph_current_survey_predictors as graph
from microcosm.build.us_runtime import graph_survey_population as survey
from microcosm.fit import qrf_target
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import WeightKind
from microcosm.graph import compile_graph, run_graph


@pytest.fixture(scope="module")
def known_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("financial-demographics")
    with pytest.MonkeyPatch.context() as patch:
        arguments = _demographic_arguments(root, patch, zero=False)
        live = survey.run_authenticated_survey_population(
            **arguments, store_root=root / "store", clones=True, return_values=True
        )
        yield live
        values.source.verify_survey_population_preparation(live.preparation)


def _qualify(live, **kwargs):
    return values.qualify_current_survey_predictors(
        live.preparation,
        live.allocated_population,
        live.clone_population,
        **kwargs,
    )


def test_demographic_features_fit_attach_and_require_replay(known_source):
    live = known_source
    original = live.clone_population
    legacy = _qualify(live)
    qualified = _qualify(live, demographic_conditioning=True)
    matrix = graph.model_input.decode_recipient_matrix(qualified.matrix)
    assert tuple(matrix.features) == values.DEMOGRAPHIC_FEATURES
    assert tuple(graph.model_input.decode_recipient_matrix(legacy.matrix).features) == (
        values.FEATURES
    )
    assert len(values.DEMOGRAPHIC_FEATURES) == 5
    assert not any("prior" in name for name in values.DEMOGRAPHIC_FEATURES)
    pd.testing.assert_frame_equal(qualified.native_money, legacy.native_money)
    details = qualified.evidence["model_judgments"]["demographic_conditioning"]
    assert details["asec_sex_and_state_observation_year"] == 2025
    assert details["acs_sex_and_state_observation_year"] == 2024
    assert details["state_is_income_year_residence_claim"] is False
    # The maintained fixture reverses ASEC source CSV rows before issuance.
    # These independently expected native IDs therefore test the keyed join.
    origins = qualified.origins
    asec_ids = origins.loc[origins.source.eq("asec")].sort_values("native_person_id")
    assert asec_ids.native_person_id.tolist() == [105, 106, 107, 108]
    donor = qualified.donor_columns.loc[asec_ids.index]
    np.testing.assert_array_equal(donor.survey_predictor_is_female, [0, 1, 0, 1])
    np.testing.assert_array_equal(donor.survey_predictor_state_fips, [6, 6, 36, 36])
    assert matrix.features.survey_predictor_is_female.eq(0).all()
    assert matrix.features.survey_predictor_state_fips.eq(6).all()
    assert qualified.donor_frame.resolve_weights("person").kind is WeightKind.DESIGN
    np.testing.assert_array_equal(
        qualified.donor_frame.resolve_weights("person").values,
        legacy.donor_frame.resolve_weights("person").values,
    )
    pins = {}
    for edge in graph.host.current_survey_host_edges():
        receipt = live.manifest.node(edge.producer)
        key = receipt.opaque_artifacts[edge.artifact]
        pins[edge.name] = {
            "producer_key": receipt.key,
            "artifact_key": key,
            "payload_sha256": graph.codec.sha(live.store.load_bytes(key)),
        }
    nodes = graph.current_survey_predictor_nodes(
        qualified, original.frame, host_pins=pins, n_estimators=2
    )
    compiled = compile_graph(
        replace(live.compiled.graph, nodes=(*live.compiled.graph.nodes, *nodes))
    )
    for cls in (
        graph.CurrentSurveyPredictorProjectionKernel,
        graph.CurrentSurveyPredictorDonorFilterKernel,
        graph.CurrentSurveyPredictorDonorColumnsKernel,
        graph.CurrentSurveyPredictorAttachKernel,
    ):
        live.kernels.register(
            cls(
                live.preparation,
                live.allocated_population,
                original,
                host_pins=pins,
                n_estimators=2,
                demographic_conditioning=True,
            )
        )
    live.kernels.register(LegacyQRFTrainKernel())
    live.kernels.register(LegacyQRFApplyMatrixKernel())
    artifacts = [
        (
            graph.PROJECTION_NODE,
            "projection",
            graph.PROJECTION_TYPE,
            graph.CurrentSurveyPredictorProjectionKernel,
        ),
        (
            graph.PROJECTION_NODE,
            "matrix",
            graph.model_input.RECIPIENT_MATRIX_TYPE,
            graph.CurrentSurveyPredictorProjectionKernel,
        ),
    ]
    for i in range(len(values.TARGETS)):
        artifacts.extend(
            (
                (
                    f"{graph.FIT_PREFIX}.{i:03d}",
                    "model",
                    qrf_target.LEGACY_QRF_TARGET_TYPE,
                    LegacyQRFTrainKernel,
                ),
                (
                    f"{graph.APPLY_PREFIX}.{i:03d}",
                    "raw_draw",
                    graph.codec.RAW_TARGET_TYPE,
                    LegacyQRFApplyMatrixKernel,
                ),
                (
                    f"{graph.APPLY_PREFIX}.{i:03d}",
                    "apply_state",
                    graph.MATRIX_APPLY_STATE_TYPE,
                    LegacyQRFApplyMatrixKernel,
                ),
            )
        )
    results = []
    for resume in ("auto", "require"):
        observed = {}
        manifest = run_graph(
            compiled,
            store=live.store,
            kernels=live.kernels,
            sources=live.sources,
            resume=resume,
            _population_observer=lambda name, population, observed=observed: (
                observed.__setitem__(name, population)
            ),
        )
        loaded = {}
        for node_id, name, type_, cls in artifacts:
            receipt = manifest.node(node_id)
            payload = live.store.load_bytes(receipt.opaque_artifacts[name])
            survey._final_artifact(
                manifest,
                live.store,
                node_id=node_id,
                name=name,
                type_=type_,
                payload=payload,
                capabilities=cls.capabilities,
            )
            loaded[node_id, name] = payload
        for i, target in enumerate(values.TARGETS):
            payload = loaded[f"{graph.FIT_PREFIX}.{i:03d}", "model"]
            model = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
                payload, expected_sha256=graph.codec.sha(payload)
            )
            assert model.target == target
            assert model._target_model.columns == (
                *values.DEMOGRAPHIC_FEATURES,
                *values.TARGETS[:i],
            )
        final = observed[graph.ATTACH_NODE]
        verification = {
            "population": final,
            "projection": loaded[graph.PROJECTION_NODE, "projection"],
            "matrix": loaded[graph.PROJECTION_NODE, "matrix"],
            "matrix_producer_key": manifest.node(graph.PROJECTION_NODE).key,
            "raw_draws": tuple(
                loaded[f"{graph.APPLY_PREFIX}.{i:03d}", "raw_draw"] for i in range(3)
            ),
            "apply_states": tuple(
                loaded[f"{graph.APPLY_PREFIX}.{i:03d}", "apply_state"] for i in range(3)
            ),
            "host_pins": pins,
            "n_estimators": 2,
            "demographic_conditioning": True,
        }
        result = graph.verify_materialized_current_survey_predictors(
            live.preparation, live.allocated_population, original, **verification
        )
        assert result["all_output_cells_available"]
        assert not result["release_eligible"]
        with pytest.raises(ValueError, match="MATERIALIZED_SOURCE"):
            graph.verify_materialized_current_survey_predictors(
                live.preparation,
                live.allocated_population,
                original,
                **{**verification, "demographic_conditioning": False},
            )
        for entity in original.frame.entities:
            before, after = original.frame.table(entity), final.frame.table(entity)
            retained = [
                c for c in before if entity != "person" or c not in values.OUTPUTS
            ]
            pd.testing.assert_frame_equal(
                before[retained], after[retained], check_exact=True
            )
        assert final.version == original.version
        assert final.weight_kind == original.weight_kind
        assert final.mass_ledger == original.mass_ledger
        for entity in original.design_weights:
            np.testing.assert_array_equal(
                final.design_weights[entity], original.design_weights[entity]
            )
            np.testing.assert_array_equal(
                final.frame.weights_for(entity).values,
                original.frame.weights_for(entity).values,
            )
        expected_owners = dict(original.owners)
        expected_owners.update(
            {("person", c): graph.ATTACH_NODE for c in values.OUTPUTS}
        )
        assert dict(final.owners) == expected_owners
        if resume == "require":
            assert all(record.hit for record in manifest.nodes.values())
        results.append(final)
    graph.replay.same_replayed_population(*results)


def test_unresolved_source_demographics_refuse_opt_in_and_preserve_default(
    tmp_path, monkeypatch
):
    arguments = _demographic_arguments(tmp_path, monkeypatch, unknown=True)
    live = survey.run_authenticated_survey_population(
        **arguments, store_root=tmp_path / "store", clones=True, return_values=True
    )
    original = values.source._frame_identity(live.preparation.frame)
    default = _qualify(live)
    assert not default.demographic_conditioning
    with pytest.raises(
        ValueError, match="CURRENT_SURVEY_PREDICTOR_DEMOGRAPHIC_UNKNOWN"
    ):
        _qualify(live, demographic_conditioning=True)
    assert values.source._frame_identity(live.preparation.frame) == original
    assert _qualify(live).projection == default.projection
    source_values = (
        values.observed_geography.demographics.qualify_current_asec_demographics(
            live.preparation
        )
    )
    assert source_values.person.is_female.isna().any()
    assert source_values.household.state_fips.isna().any()
    assert json.loads(source_values.receipt)["sex_unknown_persons"] == 1


def test_last_money_owner_borrow_cannot_change_detached_demographic_values(
    known_source,
):
    live = known_source
    captured, changed = [], []
    parent = values.source.asec_native._ISSUED[
        id(live.preparation._checked()[2].native[1])
    ][2].parent
    ready_code = parent.ready.__code__

    def profile(frame, event, arg):
        if event != "return":
            return
        if frame.f_code is values._demographic_features.__code__ and arg is not None:
            captured.append(arg[2])
        elif frame.f_code is ready_code and captured and not changed:
            table = captured[-1].person
            table.loc[table.index[0], "is_female"] = not bool(table.iloc[0].is_female)
            changed.append(True)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="DEMOGRAPHIC_PROJECTION_CHANGED"):
            _qualify(live, demographic_conditioning=True)
    finally:
        sys.setprofile(previous)
    assert changed == [True]
    # Only a detached projection changed; the live source can still qualify.
    assert _qualify(live, demographic_conditioning=True).demographic_conditioning
