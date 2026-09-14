"""Nineteen actual graph nodes over invented, source-issued survey fixtures."""

import hashlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments
from test_us_graph_atomic_survey_population import _support_payload

from microcosm.build.us_runtime import graph_atomic_survey_financial as runner
from microcosm.graph import ArtifactType, ArtifactValue, NumericScope
from microcosm.graph.keys import opaque_artifact_key

financial = runner.financial
values = financial.values


@pytest.fixture(scope="module")
def known_financial_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("atomic-survey-financial")
    with pytest.MonkeyPatch.context() as patch:
        arguments = _demographic_arguments(root, patch, unknown=False, zero=False)
        payload, source_ids = _support_payload()
        support_path = root / "invented-block-support.npz"
        support_path.write_bytes(payload)
        config = runner.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support_path),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        call = {
            **arguments,
            "store_root": root / "store",
            "geography_config": config,
            "demographic_conditioning": True,
            "n_estimators": 2,
            "return_values": True,
        }
        cold = runner.run_atomic_survey_financial(**call, resume="auto")
        warm = runner.run_atomic_survey_financial(**call, resume="require")
        yield SimpleNamespace(call=call, cold=cold, warm=warm)
        for run in (cold, warm):
            values.source.verify_survey_population_preparation(run.prefix.preparation)


def test_nineteen_node_financial_cold_and_required_replay(known_financial_run):
    case = known_financial_run
    assert case.cold.manifest.key == case.warm.manifest.key
    assert all(n.hit for n in case.warm.manifest.nodes.values())
    for run in (case.cold, case.warm):
        assert len(run.compiled.order) == 19
        edge = financial._geography_edge()
        assert edge.producer in run.compiled.predecessors[financial.DONOR_NODE]
        gate = run.manifest.node(edge.producer)
        assert gate.opaque_artifacts[edge.artifact] == opaque_artifact_key(
            gate.key, edge.artifact
        )
        assert run.store.load_bytes(
            gate.opaque_artifacts[edge.artifact]
        ) == financial.codec.encode_json(
            financial.codec.decode_json(run.projection)["atomic_geography"][
                "validation_receipt"
            ]
        )
        assert run.prefix.clone_population is run.prefix.geography_population
        assert "census_block_geoid" not in run.prefix.expanded_population.frame.table(
            "household"
        )
        assert set(run.compiled.order) == {
            *run.prefix.compiled.order,
            financial.PROJECTION_NODE,
            financial.DONOR_NODE,
            financial.DONOR_COLUMNS_NODE,
            *(f"{financial.FIT_PREFIX}.{i:03d}" for i in range(3)),
            *(f"{financial.APPLY_PREFIX}.{i:03d}" for i in range(3)),
            financial.ATTACH_NODE,
        }
        before, after = run.prefix.clone_population, run.financial_population
        assert before is not after
        assert "census_block_geoid" not in run.prefix.allocated_population.frame.table(
            "household"
        )
        assert "census_block_geoid" in before.frame.table("household")
        assert (
            tuple(financial.model_input.decode_recipient_matrix(run.matrix).features)
            == values.DEMOGRAPHIC_FEATURES
        )
        evidence = financial.codec.decode_json(run.projection)
        assert (
            evidence["model_judgments"]["demographic_conditioning"][
                "asec_sex_and_state_observation_year"
            ]
            == 2025
        )
        assert evidence["atomic_geography"]["config_sha256"] == financial.codec.sha(
            run.prefix.geography_config.to_bytes()
        )
        assert evidence["release_eligible"] is False
        assert not any("prior" in c for c in values.DEMOGRAPHIC_FEATURES)
        assert len(values.OUTPUTS) == 8
        for entity in before.frame.entities:
            retained = [
                c
                for c in before.frame.table(entity)
                if entity != "person" or c not in values.OUTPUTS
            ]
            pd.testing.assert_frame_equal(
                after.frame.table(entity)[retained],
                before.frame.table(entity)[retained],
                check_exact=True,
            )
        assert after.version == before.version
        assert after.weight_kind == before.weight_kind
        assert after.mass_ledger == before.mass_ledger
        assert after.frame.metadata == before.frame.metadata
        assert after.frame.mass_log == before.frame.mass_log
        assert after.frame.schema == before.frame.schema
        assert after.frame.links == before.frame.links
        pd.testing.assert_series_equal(
            after.frame.strata, before.frame.strata, check_exact=True
        )
        assert dict(after.owners) == {
            **before.owners,
            **{("person", c): financial.ATTACH_NODE for c in values.OUTPUTS},
        }
        for entity in before.design_weights:
            np.testing.assert_array_equal(
                after.design_weights[entity], before.design_weights[entity]
            )
            np.testing.assert_array_equal(
                after.frame.weights_for(entity).values,
                before.frame.weights_for(entity).values,
            )
        # Source-origin pairing must carry each modeled ACS draw to both arms.
        person = after.frame.person
        source_id = values.provenance.support_source_id_column("person")
        for _, rows in person.groupby(source_id, sort=False):
            assert len(rows) == 2
            for column in values.OUTPUTS:
                assert rows[column].notna().all()
                assert rows[column].nunique() == 1
        runner.survey._same_frame(after.frame, run.manifest.population(after.version))
    runner.atomic.same_replayed_population(
        case.cold.financial_population, case.warm.financial_population
    )


def test_geography_option_preserves_default_three_features(known_financial_run):
    run = known_financial_run.cold
    prefix = run.prefix
    default = values.qualify_current_survey_predictors(
        prefix.preparation,
        prefix.allocated_population,
        prefix.clone_population,
        geography_config=prefix.geography_config,
    )
    opted = values.qualify_current_survey_predictors(
        prefix.preparation,
        prefix.allocated_population,
        prefix.clone_population,
        geography_config=prefix.geography_config,
        demographic_conditioning=True,
    )
    assert default.demographic_conditioning is False
    assert (
        tuple(financial.model_input.decode_recipient_matrix(default.matrix).features)
        == values.FEATURES
    )
    assert len(values.FEATURES) == 3
    assert len(values.DEMOGRAPHIC_FEATURES) == 5
    pd.testing.assert_frame_equal(
        default.native_money, opted.native_money, check_exact=True
    )
    assert (
        default.geography_config_payload
        == opted.geography_config_payload
        == prefix.geography_config.to_bytes()
    )
    # A geography-enriched clone cannot be admitted through the historical arm.
    with pytest.raises(ValueError, match="CLONE_FRAME_VALUES"):
        values.qualify_current_survey_predictors(
            prefix.preparation, prefix.allocated_population, prefix.clone_population
        )


def test_final_support_return_cannot_mutate_detached_money(known_financial_run):
    case, fired = known_financial_run, []

    def profile(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code is runner.reconstruction._read_support.__code__
            and caller is not None
            and caller.f_code is values.qualify_current_survey_predictors.__code__
            and "result" in caller.f_locals
            and not fired
        ):
            fired.append(True)
            result = caller.f_locals["result"]
            index = result.origins.index[result.origins.source.eq("asec")][0]
            result.native_money.loc[index, "WSAL_VAL"] += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        prefix = case.cold.prefix
        with pytest.raises(ValueError, match="FINAL_DERIVED_VALUES"):
            values.qualify_current_survey_predictors(
                prefix.preparation,
                prefix.allocated_population,
                prefix.clone_population,
                geography_config=prefix.geography_config,
                demographic_conditioning=True,
            )
    finally:
        sys.setprofile(previous)
    assert fired == [True]


def test_final_owner_return_cannot_mutate_materialized_geography(known_financial_run):
    case, fired = known_financial_run, []

    def wrapper_profile(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is financial.verify_materialized_current_survey_predictors.__code__
            and caller is not None
            and caller.f_code is runner.run_atomic_survey_financial.__code__
            and "result" in caller.f_locals
            and not fired
        ):
            fired.append(True)
            table = caller.f_locals["result"].financial_population.frame.table(
                "household"
            )
            table.loc[table.index[0], "survey_observed_state"] = "99"

    previous = sys.getprofile()
    sys.setprofile(wrapper_profile)
    try:
        with pytest.raises(ValueError, match="ATOMIC_FINAL_POPULATION_MUTATION"):
            runner.run_atomic_survey_financial(**case.call, resume="require")
    finally:
        sys.setprofile(previous)
    assert fired == [True]


@pytest.mark.parametrize(
    "defect,reason",
    (
        ("payload", "GEOGRAPHY_VALIDATION_BINDING"),
        ("producer", "GEOGRAPHY_VALIDATION_BINDING"),
        ("type", "DETAIL_ARTIFACT_IDENTITY"),
    ),
)
def test_financial_donor_requires_actual_geography_gate(
    known_financial_run, defect, reason
):
    run = known_financial_run.cold
    node = run.compiled.graph.node(financial.DONOR_NODE)
    artifacts = {}
    for edge in node.artifact_inputs:
        producer = run.manifest.node(edge.producer)
        key = producer.opaque_artifacts[edge.artifact]
        artifacts[edge.name] = ArtifactValue(
            run.store.load_bytes(key), edge.type, key, producer.key, NumericScope()
        )
    # A detached consumer-context negative, using actual source-owned run keys.
    # This does not pretend that constructing ArtifactValue issues an authority.
    edge = financial._geography_edge()
    original = artifacts[edge.name]
    if defect == "payload":
        artifacts[edge.name] = replace(original, payload=original.payload + b" ")
    elif defect == "producer":
        other = "0" * 64 if original.producer_key != "0" * 64 else "1" * 64
        artifacts[edge.name] = replace(
            original, producer_key=other, key=opaque_artifact_key(other, edge.artifact)
        )
    else:
        artifacts[edge.name] = replace(
            original, type=ArtifactType("invented.wrong_gate", 1)
        )
    context = SimpleNamespace(node=node, sources={}, artifacts=artifacts)
    kernel = run.kernels.get(financial.CurrentSurveyPredictorDonorFilterKernel.ref)
    with pytest.raises(ValueError, match=reason):
        kernel._qualified(context)
