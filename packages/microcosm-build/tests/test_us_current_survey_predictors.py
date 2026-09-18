"""Actual source owners and graph replay on pinned invented survey inputs.

No production source issuer/readiness call is replaced. The maintained fixture
constructs its own real bytes and test pins before normal authentication. The two fixtures exercise both zero and nonconstant positive-regime forests. Neither
fixture establishes held-out fit quality.
"""

import copy
import json
import pickle
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_survey_puf_transfer import _copy_population
from test_us_graph_survey_population import authenticated_arguments

from microcosm.build.us_runtime import cps_carried_current as leaves
from microcosm.build.us_runtime import current_survey_predictors as values
from microcosm.build.us_runtime import graph_current_survey_predictors as graph
from microcosm.build.us_runtime import graph_survey_population as survey
from microcosm.build.us_runtime import puf_support as support
from microcosm.build.us_runtime import survey_population_preparation as preparation
from microcosm.fit import qrf, qrf_target
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import Frame, WeightKind
from microcosm.frame.bundle import _freeze_metadata
from microcosm.graph import compile_graph, run_graph
from microcosm.graph.keys import opaque_artifact_key


@pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
def test_metadata_copy_preserves_producer_identity(monkeypatch, operation):
    metadata = _freeze_metadata({"origin": {"stages": ["survey", "clone"]}})
    # Exercise a cold class even if a preceding test already copied metadata.
    # Python 3.14 annotation closures capture this mutable class namespace.
    monkeypatch.delattr(type(metadata), "__slotnames__", raising=False)
    namespace = dict(vars(type(metadata)))
    producer = preparation._live()

    if operation == "pickle":
        restored = pickle.loads(pickle.dumps(metadata))
    else:
        restored = getattr(copy, operation)(metadata)

    assert restored == metadata
    assert restored is not metadata
    assert restored["origin"]["stages"] == ("survey", "clone")
    with pytest.raises(TypeError):
        restored["origin"]["stages"] = ()
    assert dict(vars(type(metadata))) == namespace
    assert preparation._live() == producer


@pytest.mark.parametrize("method", ["__getitem__", "__reduce__"])
def test_metadata_method_replacement_still_invalidates_producer(monkeypatch, method):
    metadata = _freeze_metadata({"origin": "survey"})
    producer = preparation._live()
    monkeypatch.setattr(type(metadata), method, lambda *args: None)
    assert preparation._live() != producer
    with pytest.raises(
        preparation.SurveyPopulationPreparationError, match="PRODUCER_CHANGED"
    ):
        preparation._producer()


def test_survey_source_ids_preserve_zero_without_relaxing_physical_or_unique_axis():
    original = pd.Series([0, 2, 5], dtype="int64")
    result = values._ids(original)
    np.testing.assert_array_equal(result, np.array([0, 2, 5], dtype="int64"))
    result[0] = 9
    assert original.iloc[0] == 0
    for bad in (
        pd.Series([-1, 0, 2], dtype="int64"),
        pd.Series([0, 0, 2], dtype="int64"),
    ):
        with pytest.raises(ValueError, match="^CURRENT_SURVEY_PREDICTOR_ID_AXIS$"):
            values._ids(bad)
    for bad in (
        pd.Series([0.0, 1.0]),
        pd.Series([False, True]),
        pd.Series([0, 1], dtype="Int64"),
    ):
        with pytest.raises(ValueError, match="^CURRENT_SURVEY_PREDICTOR_ID_DTYPE$"):
            values._ids(bad)


def _money():
    return {
        "WSAL_VAL": np.array([0.0, 50000.0, 100000.0]),
        "SEMP_VAL": np.array([-400.0, 0.0, 500.0]),
        "INT_VAL": np.array([0.0, 120.0, 300.0]),
        "DIV_VAL": np.array([40.0, 0.0, 75.0]),
        "CAP_VAL": np.array([-700.0, 0.0, 1100.0]),
    }


def test_shared_current_money_split_preserves_signs_and_partition_totals():
    source = _money()
    result = leaves.derive_cps_current_predictor_leaves(source)
    assert set(result) == set(leaves.CPS_CURRENT_PREDICTOR_PERSON_LEAVES)
    # The legacy full leaf owner unpacks this mapping. Preserve its original
    # insertion order so consumers constructing a DataFrame see the same schema.
    assert tuple(result) == (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "taxable_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "long_term_capital_gains_before_response",
        "short_term_capital_gains",
    )
    np.testing.assert_array_equal(
        result["self_employment_income_before_lsr"], source["SEMP_VAL"]
    )
    np.testing.assert_allclose(
        result["qualified_dividend_income"] + result["non_qualified_dividend_income"],
        source["DIV_VAL"],
        rtol=1e-15,
        atol=0,
    )
    np.testing.assert_allclose(
        result["short_term_capital_gains"]
        + result["long_term_capital_gains_before_response"],
        source["CAP_VAL"],
        rtol=1e-15,
        atol=0,
    )
    np.testing.assert_array_equal(
        result["taxable_interest_income"],
        source["INT_VAL"] * leaves.TAXABLE_INTEREST_FRACTION,
    )
    result["employment_income_before_lsr"][0] = 10.0
    assert source["WSAL_VAL"][0] == 0.0


@pytest.mark.parametrize("reverse", (False, True))
def test_financial_attachment_conserves_native_and_drawn_interest(reverse):
    # Descriptive, invented inputs exercise the numerical attachment only;
    # constructing this dataclass does not authenticate a survey source.
    ids = pd.Index([11, 12, 13], name="person_id")
    money = pd.DataFrame(0.0, index=ids, columns=values.MONEY_FIELDS)
    money["INT_VAL"] = [120.031, 0.0, np.nan]
    origins = pd.DataFrame(
        {"native_person_id": [101, 102, 201], "source": ["asec", "asec", "acs"]},
        index=ids,
    )
    features = pd.DataFrame(0.0, index=ids[-1:], columns=values.FEATURES)
    qualified = values.QualifiedSurveyPredictors(
        projection=b"",
        matrix=graph.model_input.encode_recipient_matrix(
            features, entity="person", entity_ids=ids[-1:].to_numpy(dtype="<i8")
        ),
        source_frame=None,
        donor_frame=None,
        donor_columns=pd.DataFrame(),
        native_money=money.iloc[::-1] if reverse else money,
        origins=origins.iloc[::-1] if reverse else origins,
        evidence={},
    )
    # Interleave pairs and origins to catch positional attachment.
    stack_ids = [13, 11, 12, 11, 13, 12]
    people = pd.DataFrame(
        {
            "person_id": np.arange(301, 307, dtype=np.int64),
            values.provenance.support_source_id_column("person"): stack_ids,
            values.provenance.spine_source_id_column("person"): [
                201,
                101,
                102,
                101,
                201,
                102,
            ],
            values.provenance.support_channel_column("person"): [
                "acs",
                "asec",
                "asec",
                "asec",
                "acs",
                "asec",
            ],
            values.provenance.support_clone_index_column("person"): [1, 0, 1, 1, 0, 0],
        }
    )
    draws = pd.DataFrame([[19.079, 0.0, 0.0]], index=ids[-1:], columns=values.TARGETS)
    result = values.complete_predictor_columns(
        qualified, SimpleNamespace(person=people), draws
    )
    assert ("person", "tax_exempt_interest_income") in result
    taxable = result["person", "taxable_interest_income"]
    exempt = result["person", "tax_exempt_interest_income"]
    total = (
        money["INT_VAL"].fillna(draws[values.TARGETS[0]]).reindex(stack_ids).to_numpy()
    )
    np.testing.assert_array_equal(taxable, total * leaves.TAXABLE_INTEREST_FRACTION)
    np.testing.assert_array_equal(exempt, total - taxable.to_numpy())
    np.testing.assert_allclose(taxable + exempt, total, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(exempt.to_numpy()[[2, 5]], [0.0, 0.0])
    assert taxable.index.tolist() == people.person_id.tolist()
    assert exempt.index.equals(taxable.index)
    assert money.loc[11, "INT_VAL"] == 120.031
    assert np.isnan(money.loc[13, "INT_VAL"])
    assert draws.iloc[0, 0] == 19.079


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.nan, 1.0, 2.0]),
        np.array([np.inf, 1.0, 2.0]),
        np.array(["0", "1", "2"]),
        np.array([True, False, True]),
        np.array([1, 2, 3]),
        np.zeros((3, 1)),
        np.zeros(2),
    ],
)
def test_shared_money_split_rejects_missing_or_coerced_source(bad):
    source = _money()
    source["INT_VAL"] = bad
    with pytest.raises(ValueError, match="PREDICTOR_MONEY"):
        leaves.derive_cps_current_predictor_leaves(source)


@pytest.mark.parametrize("nonconstant", (False, True))
def test_current_predictor_source_fit_attachment_and_required_replay(
    tmp_path, monkeypatch, nonconstant
):
    observations = {
        "WSAL_VAL": np.array([40000.0, 0.0, 140000.0, 0.0]),
        "SEMP_VAL": np.array([1200.0, 0.0, 12000.0, 0.0]),
        "INT_VAL": np.array([120.0, 0.0, 9000.0, 0.0]),
        "DIV_VAL": np.array([400.0, 0.0, 4000.0, 0.0]),
        "CAP_VAL": np.array([500.0, 0.0, 14000.0, 0.0]),
    }
    fixture_kwargs = (
        {"zero": False, "current_predictor_money": observations} if nonconstant else {}
    )
    live = survey.run_authenticated_survey_population(
        **authenticated_arguments(tmp_path, monkeypatch, **fixture_kwargs),
        clones=True,
        return_values=True,
    )
    preparation, allocated, clone = (
        live.preparation,
        live.allocated_population,
        live.clone_population,
    )
    qualified = values.qualify_current_survey_predictors(preparation, allocated, clone)
    assert qualified.evidence["asec_interview_year"] == 2025
    assert qualified.evidence["asec_income_window"] == "calendar_year_2024"
    assert qualified.evidence["price_basis_year"] == 2024
    # The producer's actual MappingProxyType receipt stays read-only. The
    # portable projection contains an explicit plain snapshot, not a generic
    # serializer fallback that could silently coerce unknown evidence objects.
    assert type(qualified.evidence["earnings_universe"]) is dict
    assert (
        json.loads(qualified.projection)["earnings_universe"]
        == qualified.evidence["earnings_universe"]
    )
    assert (
        qualified.evidence["model_judgments"]["quality_acceptance"]
        == "held_out_fit_quality_not_yet_assessed"
    )
    assert qualified.evidence["model_judgments"]["interest_partition"] == {
        "total": "source-qualified_ASEC_INT_VAL_or_ACS_modeled_INT_VAL",
        "taxable": "INT_VAL*maintained_taxable_interest_fraction",
        "tax_exempt": "INT_VAL-taxable_interest_income",
        "split_is_observed": False,
    }
    assert qualified.donor_frame.resolve_weights("person").kind is WeightKind.DESIGN
    # A separate numerical probe changes only a defensive invented copy. It
    # exercises the actual age15 owner; it is never issued as authenticated.
    source_frame = qualified.source_frame
    probe = Frame(
        {e: source_frame.table(e).copy(deep=True) for e in source_frame.entities},
        source_frame.schema,
        {e: source_frame.weights_for(e) for e in source_frame.weighted_entities},
        source_frame.strata.copy(deep=True),
        metadata=source_frame.metadata,
        mass_log=source_frame.mass_log,
    )
    acs_index = probe.person.index[
        probe.person[values.provenance.support_channel_column("person")].eq("acs")
    ][0]
    for name in ("age", "AGEP"):
        probe.person.loc[acs_index, name] = 14
    for name in ("WAGP", "SEMP", *values.OUTPUTS[:2]):
        probe.person.loc[acs_index, name] = np.nan
    completed_probe, receipt = values._acs_earnings(probe)
    assert completed_probe.person.loc[acs_index, list(values.OUTPUTS[:2])].eq(0).all()
    assert completed_probe.person.loc[acs_index, ["WAGP", "SEMP"]].isna().all()
    assert receipt["structurally_absent_person_rows"] > 0
    assert type(receipt) is MappingProxyType
    with pytest.raises(TypeError):
        receipt["minimum_age"] = 0
    assert receipt["minimum_age"] == 15
    probe.person.loc[acs_index, "age"] = 15
    probe.person.loc[acs_index, "AGEP"] = 15
    with pytest.raises(ValueError, match="ACS_ELIGIBLE_EARNINGS_UNKNOWN"):
        values._acs_earnings(probe)
    matrix = graph.model_input.decode_recipient_matrix(qualified.matrix)
    invented_draw = pd.DataFrame(
        {
            target: np.arange(1.0, 1.0 + len(matrix.entity_ids)) * (i + 1.0)
            for i, target in enumerate(values.TARGETS)
        },
        index=matrix.features.index,
    )
    original_join = values.complete_predictor_columns(
        qualified, clone.frame, invented_draw
    )
    reversed_join = values.complete_predictor_columns(
        replace(
            qualified,
            native_money=qualified.native_money.iloc[::-1],
            origins=qualified.origins.iloc[::-1],
        ),
        clone.frame,
        invented_draw,
    )
    for name in original_join:
        pd.testing.assert_series_equal(original_join[name], reversed_join[name])
    completed_interest = qualified.native_money["INT_VAL"].fillna(
        invented_draw[values.TARGETS[0]]
    )
    expected_interest = completed_interest.reindex(
        clone.frame.person[values.provenance.support_source_id_column("person")]
    ).to_numpy()
    np.testing.assert_allclose(
        original_join["person", "taxable_interest_income"]
        + original_join["person", "tax_exempt_interest_income"],
        expected_interest,
        rtol=1e-15,
        atol=0,
    )
    assert allocated.frame.resolve_weights("person").kind is WeightKind.IMPORTANCE
    if nonconstant:
        origins = qualified.origins
        ids = origins.index[origins.source.eq("asec")]
        # Join the current-money projection by native identity, never row position.
        ordered = origins.loc[ids].sort_values("native_person_id").index
        assert origins.loc[ordered, "native_person_id"].tolist() == [105, 106, 107, 108]
        for field, expected in observations.items():
            np.testing.assert_array_equal(
                qualified.native_money.loc[ordered, field], expected
            )
        for target in values.TARGETS:
            assert qualified.donor_columns[target].nunique() == 3
        weights = qualified.donor_frame.resolve_weights("person")
        assert weights.kind is WeightKind.DESIGN and (weights.values > 0).all()
        np.testing.assert_array_equal(weights.values, [2552.12, 2552.12, 100.0, 100.0])
    else:
        assert qualified.donor_columns.loc[:, list(values.TARGETS)].eq(0).all().all()
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
        qualified, clone.frame, host_pins=pins, n_estimators=2
    )
    assert nodes[1].base == survey.CREATE_NODE
    assert nodes[0].population == graph.DONOR_NODE
    assert tuple(edge.name for edge in nodes[1].artifact_inputs) == ("preparation",)
    assert all("state" not in c and "female" not in c for c in values.FEATURES)
    compiled = compile_graph(
        replace(live.compiled.graph, nodes=(*live.compiled.graph.nodes, *nodes))
    )
    # Actual compiler dependencies retain allocation evidence without an
    # ordinary node in CREATE depending on its own downstream allocation.
    assert graph.PROJECTION_NODE not in compiled.predecessors[survey.ALLOCATION_NODE]
    assert graph.DONOR_NODE in compiled.predecessors[graph.PROJECTION_NODE]
    assert survey.ALLOCATION_NODE in compiled.predecessors[graph.PROJECTION_NODE]
    assert graph.PROJECTION_NODE not in compiled.predecessors[graph.DONOR_NODE]
    for cls in (
        graph.CurrentSurveyPredictorProjectionKernel,
        graph.CurrentSurveyPredictorDonorFilterKernel,
        graph.CurrentSurveyPredictorDonorColumnsKernel,
        graph.CurrentSurveyPredictorAttachKernel,
    ):
        live.kernels.register(
            cls(preparation, allocated, clone, host_pins=pins, n_estimators=2)
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
    for i in range(3):
        artifacts.extend(
            [
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
            ]
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
        donor_weights = observed[graph.DONOR_NODE].frame.resolve_weights("person")
        assert donor_weights.kind is WeightKind.DESIGN
        np.testing.assert_array_equal(
            donor_weights.values,
            qualified.donor_frame.resolve_weights("person").values,
        )
        loaded = {}
        for node_id, name, type_, cls in artifacts:
            receipt = manifest.node(node_id)
            key = opaque_artifact_key(receipt.key, name)
            assert receipt.opaque_artifacts[name] == key
            payload = live.store.load_bytes(key)
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
            fit_id = f"{graph.FIT_PREFIX}.{i:03d}"
            payload = loaded[fit_id, "model"]
            # Actual typed producer/store verification above precedes pickle loading.
            artifact = qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
                payload, expected_sha256=graph.codec.sha(payload)
            )
            assert artifact.target == target
            assert artifact._target_model.columns == (
                *values.FEATURES,
                *values.TARGETS[:i],
            )
            expected_regime = (
                qrf.Regime.ZERO_INFLATED_POSITIVE
                if nonconstant
                else qrf.Regime.DEGENERATE_ZERO
            )
            assert artifact.regime == expected_regime
            if nonconstant:
                assert artifact._target_model.gate is not None
                assert artifact._target_model.positive is not None
        if resume == "require":
            assert all(record.hit for record in manifest.nodes.values())
        raw = tuple(
            loaded[f"{graph.APPLY_PREFIX}.{i:03d}", "raw_draw"] for i in range(3)
        )
        states = tuple(
            loaded[f"{graph.APPLY_PREFIX}.{i:03d}", "apply_state"] for i in range(3)
        )
        matrix_key = manifest.node(graph.PROJECTION_NODE).key
        actual = observed[graph.ATTACH_NODE]
        verification = dict(
            population=actual,
            projection=loaded[graph.PROJECTION_NODE, "projection"],
            matrix=loaded[graph.PROJECTION_NODE, "matrix"],
            matrix_producer_key=matrix_key,
            raw_draws=raw,
            apply_states=states,
            host_pins=pins,
            n_estimators=2,
        )
        result = graph.verify_materialized_current_survey_predictors(
            preparation, allocated, clone, **verification
        )
        assert (
            result["all_output_cells_available"] is True
            and result["release_eligible"] is False
        )
        assert not actual.frame.person.loc[:, list(values.OUTPUTS)].isna().any().any()
        actual_person = actual.frame.person
        for _, rows in actual_person.groupby(
            values.provenance.support_source_id_column("person")
        ):
            assert len(rows) == 2
            np.testing.assert_array_equal(
                rows.loc[:, list(values.OUTPUTS)].iloc[0],
                rows.loc[:, list(values.OUTPUTS)].iloc[1],
            )
        for raw_name in ("WAGP", "SEMP", "ADJINC"):
            pd.testing.assert_series_equal(
                actual_person[raw_name], clone.frame.person[raw_name]
            )
        puf_mask = support.puf_tax_detail_clone_mask(
            actual.frame.table("tax_unit"), entity="tax_unit"
        )
        features, _ = support._strict_recipient_predictor_surface(
            actual.frame,
            puf_mask,
            support.PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
            person_outputs=(),
        )
        assert features.shape[1] == 8 and np.isfinite(features.to_numpy()).all()
        changed = _copy_population(actual)
        changed.frame.person.loc[changed.frame.person.index[0], values.OUTPUTS[2]] += (
            1.0
        )
        with pytest.raises(ValueError):
            graph.verify_materialized_current_survey_predictors(
                preparation, allocated, clone, **{**verification, "population": changed}
            )
        with pytest.raises(ValueError, match="DRAW_MATRIX_BINDING"):
            graph.read_current_survey_draws(qualified.matrix, "f" * 64, raw, states)
        results.append(actual)
    graph.replay.same_replayed_population(*results)
