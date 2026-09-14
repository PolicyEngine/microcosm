"""PUF55 controls with invented measurements and actual QRF graph execution.

No native source admission, Social Security model, or release claim.
"""

import numpy as np
import pandas as pd
import pytest
from test_us_full_puf_enrichment import InventedSource
from test_us_full_puf_output_profiles import (
    InventedProfileMatrix,
    _puf59_columns,
    _puf59_frame,
    _puf59_return_only_columns,
)

from microcosm.build.us_runtime import full_puf_enrichment as full
from microcosm.build.us_runtime import puf_support as support
from microcosm.fit import model_input
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    ContentStore,
    Graph,
    KernelRegistry,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import load_frame_store
from microcosm.graph.keys import opaque_artifact_key

PROFILE = full.PUF55_SURVEY_SS
SS_TOTAL = full.SURVEY_SS_TOTAL_PREDICTOR
SS = full.SURVEY_SS_COMPONENTS


def _columns():
    p, t, pk, tk = _puf59_columns()
    t[SS_TOTAL] = [0.0, 100.0, 1200.0, 24000.0]
    tk[SS_TOTAL] = True
    return p.drop(columns=list(SS)), t, pk.drop(columns=list(SS)), tk


def _donor55():
    p, t, pk, tk = _columns()
    return full.canonical_full_puf_donor(
        p, t, person_known=pk, tax_unit_known=tk, profile=PROFILE
    )


def _frame():
    frame = _puf59_frame()
    t = frame.table("tax_unit")
    t[SS_TOTAL] = np.resize(np.array([0.0, 0.0, 1200.0, 24000.0]), len(t))
    for j, name in enumerate(SS):
        frame.person[name] = np.resize(
            np.array([-0.0, np.nan, 30.0 + j, 0.0]), len(frame.person)
        )
    frame.revalidate()
    return frame


def _known55(frame):
    t = frame.table("tax_unit")
    mask = support.puf_tax_detail_clone_mask(t, entity="tax_unit")
    return pd.DataFrame(
        True,
        columns=PROFILE.predictors,
        index=pd.Index(t.loc[mask, "tax_unit_id"].to_numpy(), name="tax_unit_id"),
    )


def test_profile_is_explicit_and_leaves_historical_contracts_unchanged():
    assert len(PROFILE.targets) == 55 and len(PROFILE.person_outputs) == 52
    assert PROFILE.predictors == (*full.PUF59.predictors, SS_TOTAL)
    assert PROFILE.source_predictors == (*full.PUF59.source_predictors, SS_TOTAL)
    assert not set(SS) & set(PROFILE.targets)
    assert not set(full.SCF_MORTGAGE_OUTPUTS) & set(PROFILE.targets)
    assert len(full.FULL65.targets) == 65 and len(full.PUF59.targets) == 59
    assert (
        full.FULL65.person_outputs == full.PUF59.person_outputs == full.PERSON_OUTPUTS
    )
    assert len({PROFILE.phase, full.FULL65.phase, full.PUF59.phase}) == 3
    assert full.FULL65.phase == full.PHASE
    assert full.PUF59.phase == full.PHASE + ".puf59"


def test_donor_total_is_an_explicit_measurement_and_ss_components_are_not_consumed():
    p, t, pk, tk = _columns()
    for name in SS:
        p[name] = ["not an observed component"] * len(p)
    result = full.canonical_full_puf_donor(
        p, t, person_known=pk, tax_unit_known=tk, profile=PROFILE
    )
    assert tuple(result) == (
        *PROFILE.predictors,
        *PROFILE.targets,
        "weight",
        "puf_person_incidence_capacity",
    )
    np.testing.assert_array_equal(result[SS_TOTAL], t[SS_TOTAL])
    assert result.weight.tolist() == [1.0, 0.0, 2.0, 3.0]
    assert not set(SS) & set(result)


def test_return_only_profile_has_no_synthetic_persons_or_ss_output_carriers():
    returns, known = _puf59_return_only_columns()
    returns = returns.drop(columns=list(SS))
    known = known.drop(columns=list(SS))
    returns[SS_TOTAL] = [0.0, 100.0, 1200.0, 24000.0]
    known[SS_TOTAL] = True
    result = full.canonical_full_puf_donor(
        None,
        returns,
        person_known=None,
        tax_unit_known=known,
        person_targets_at_tax_unit=PROFILE.person_outputs,
        profile=PROFILE,
    )
    assert len(result) == 4
    assert result.puf_person_incidence_capacity.eq(1.0).all()
    assert not set(SS) & set(result)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, "0", True])
def test_invalid_total_is_never_synthesized_from_components(value):
    p, t, pk, tk = _columns()
    t[SS_TOTAL] = t[SS_TOTAL].astype(object)
    t.loc[0, SS_TOTAL] = value
    with pytest.raises(
        ValueError,
        match="PUF_(PROFILE_SOCIAL_SECURITY_DOMAIN|UNKNOWN|NONFINITE|PHYSICAL_TYPE)",
    ):
        full.canonical_full_puf_donor(
            p, t, person_known=pk, tax_unit_known=tk, profile=PROFILE
        )


def test_missing_or_unknown_donor_total_refuses():
    p, t, pk, tk = _columns()
    with pytest.raises(ValueError, match="PUF_DONOR_COLUMNS:tax_unit"):
        full.canonical_full_puf_donor(
            p,
            t.drop(columns=SS_TOTAL),
            person_known=pk,
            tax_unit_known=tk,
            profile=PROFILE,
        )
    tk.loc[0, SS_TOTAL] = False
    with pytest.raises(ValueError, match="PUF_UNKNOWN:tax_unit"):
        full.canonical_full_puf_donor(
            p, t, person_known=pk, tax_unit_known=tk, profile=PROFILE
        )


def test_recipient_total_uses_declared_tax_unit_grain_and_knownness():
    f, donor = _frame(), _donor55()
    k = _known55(f)
    result = full.prepare_full_puf_inputs(f, donor, predictor_known=k, profile=PROFILE)
    matrix = model_input.decode_recipient_matrix(result.matrix)
    assert len(matrix.features.columns) == 9
    np.testing.assert_array_equal(
        matrix.features[SS_TOTAL], [0.0, 0.0, 1200.0, 24000.0]
    )
    k.iloc[0, -1] = False
    with pytest.raises(ValueError, match="PUF_UNKNOWN:recipient_predictor"):
        full.prepare_full_puf_inputs(f, donor, predictor_known=k, profile=PROFILE)
    k.iloc[0, -1] = True
    del f.table("tax_unit")[SS_TOTAL]
    with pytest.raises(ValueError, match="PUF_PROFILE_PREDICTOR_SOURCE"):
        full.prepare_full_puf_inputs(f, donor, predictor_known=k, profile=PROFILE)


def test_real_puf55_chain_preserves_survey_ss_and_replays(tmp_path):
    frame, donor = _frame(), _donor55()
    known = _known55(frame)
    prepared = full.prepare_full_puf_inputs(
        frame, donor, predictor_known=known, profile=PROFILE
    )
    matrix = model_input.decode_recipient_matrix(prepared.matrix)
    # An explicit minimal recipient model Frame for the ordinary graph control;
    # finalization below consumes the actual retained US clone Frame.
    table = matrix.features.copy()
    table.insert(0, "tax_unit_id", matrix.entity_ids)
    from microcosm.frame import EntitySchema

    recipient = Frame(
        {
            "tax_unit": table,
            "person": pd.DataFrame(
                {
                    "person_id": matrix.entity_ids,
                    "person_tax_unit_id": matrix.entity_ids,
                }
            ),
        },
        EntitySchema(group_entities=("tax_unit",)),
        {
            "tax_unit": Weights(
                frame.resolve_weights("tax_unit").values[
                    support.puf_tax_detail_clone_mask(
                        frame.table("tax_unit"), entity="tax_unit"
                    )
                ],
                WeightKind.DESIGN,
            )
        },
    )
    store = ContentStore(tmp_path / "store", codecs={"frame-store": load_frame_store})
    sources, nodes = {}, []
    for name, value in (("donor", prepared.donor_frame), ("recipient", recipient)):
        key = full.codec.sha(("invented-full-puf:" + name).encode())
        sources[name] = store.put_frame(key, value)
        nodes.append(
            Node(
                name,
                InventedSource.ref,
                sources=(name,),
                structural=StructuralDelta.CREATE,
                outputs=tuple(
                    Owned("tax_unit", col, "float64")
                    for col in value.table("tax_unit")
                    if col != "tax_unit_id"
                ),
            )
        )
    fits, applies = full.full_puf_train_apply_nodes(
        donor_population="donor",
        recipient_population="recipient",
        matrix_producer="matrix",
        seed=578,
        n_estimators=2,
        zero_atol=0,
        profile=PROFILE,
    )
    assert len(fits) == len(applies) == 55
    assert [
        e.name for e in applies[-1].artifact_inputs if e.name.startswith("prior_")
    ] == [f"prior_{i:03d}" for i in range(54)]
    nodes.append(
        Node(
            "matrix",
            InventedProfileMatrix.ref,
            population="recipient",
            inputs=(Slice("tax_unit", PROFILE.predictors),),
            artifact_outputs=(
                ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
            ),
        )
    )
    compiled = compile_graph(
        Graph(
            "us",
            tuple(SourceRef(name, "frame-store") for name in sources),
            (*nodes, *fits, *applies),
        )
    )
    registry = KernelRegistry()
    for kernel in (
        InventedSource(),
        InventedProfileMatrix(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        registry.register(kernel)
    actual = run_graph(compiled, sources=sources, store=store, kernels=registry)

    def payload(manifest, node_id, artifact):
        record = manifest.node(node_id)
        key = opaque_artifact_key(record.key, artifact)
        assert key == record.opaque_artifacts[artifact]
        return store.load_bytes(key)

    raw = {
        target: payload(actual, node.id, "raw_draw")
        for target, node in zip(PROFILE.targets, applies, strict=True)
    }
    kwargs = dict(
        matrix=payload(actual, "matrix", "matrix"),
        matrix_producer_key=actual.node("matrix").key,
        raw_draws=raw,
        apply_state=payload(actual, applies[-1].id, "apply_state"),
        training_state=payload(actual, fits[-1].id, "training_state"),
        seed=578,
        profile=PROFILE,
    )
    assert kwargs["matrix"] == prepared.matrix
    before = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    last_model = payload(actual, fits[-1].id, "model")
    result, receipt = full.finalize_full_puf(
        frame, donor, predictor_known=known, last_model=last_model, **kwargs
    )
    assert receipt["release_eligible"] is False and receipt["tail_bounds"]
    assert receipt["target_order"] == list(PROFILE.targets)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(frame.table(entity), before[entity])
    np.testing.assert_array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )
    for entity, outputs in (
        ("person", PROFILE.person_outputs),
        ("tax_unit", PROFILE.tax_unit_outputs),
    ):
        mask = support.puf_tax_detail_clone_mask(result.table(entity), entity=entity)
        for output in outputs:
            assert result.table(entity).loc[mask, output].notna().all()
            if output not in before[entity]:
                assert result.table(entity).loc[~mask, output].isna().all()
            else:
                actual_cells = result.table(entity).loc[~mask, output]
                expected_cells = before[entity].loc[~mask, output]
                # The full finalizer materializes requested Boolean outputs
                # with nullable storage. Compare exact native values and NA
                # semantics in that declared Boolean representation. The SS
                # columns below have separate unnormalized bit-pattern checks.
                if pd.api.types.is_bool_dtype(expected_cells.dtype):
                    assert pd.api.types.is_bool_dtype(actual_cells.dtype)
                    actual_cells = actual_cells.astype("boolean")
                    expected_cells = expected_cells.astype("boolean")
                pd.testing.assert_series_equal(
                    actual_cells,
                    expected_cells,
                    check_exact=True,
                )
    for component in SS:
        np.testing.assert_array_equal(
            result.person[component].to_numpy().view("uint64"),
            before["person"][component].to_numpy().view("uint64"),
        )
    np.testing.assert_array_equal(
        result.table("tax_unit")[SS_TOTAL].to_numpy().view("uint64"),
        before["tax_unit"][SS_TOTAL].to_numpy().view("uint64"),
    )
    changed_frame = _frame()
    tax = changed_frame.table("tax_unit")
    tax.loc[tax.index[-1], SS_TOTAL] += 1.0
    with pytest.raises(ValueError, match="PUF_RECIPIENT_MATRIX_CHANGED"):
        full.finalize_full_puf(
            changed_frame, donor, predictor_known=known, last_model=last_model, **kwargs
        )
    people = result.table("person")
    child = support.puf_tax_detail_clone_mask(people, entity="person") & people.age.lt(
        15
    )
    assert people.loc[child, "employment_income_before_lsr"].eq(0).all()
    changed_raw = dict(raw)
    values = full.codec.read_raw_target(
        raw[PROFILE.targets[0]], target=PROFILE.targets[0], index=matrix.features.index
    ).copy()
    values[0] += 1.0
    changed_raw[PROFILE.targets[0]] = full.codec.encode_raw_target(
        values, target=PROFILE.targets[0], index=matrix.features.index
    )
    with pytest.raises(ValueError, match="PUF_RAW_HISTORY"):
        full.decode_full_puf_draws(**{**kwargs, "raw_draws": changed_raw})
    with pytest.raises(ValueError, match="PUF_FULL_MATRIX_BINDING"):
        full.decode_full_puf_draws(**{**kwargs, "matrix_producer_key": "f" * 64})
    changed_donor = donor.copy(deep=True)
    changed_donor.loc[0, "casualty_loss"] += 1.0
    with pytest.raises(ValueError, match="PUF_DONOR_CONSUMED_BYTES"):
        full.finalize_full_puf(
            frame, changed_donor, predictor_known=known, last_model=last_model, **kwargs
        )
    incomplete = dict(raw)
    incomplete.pop(PROFILE.targets[-1])
    with pytest.raises(ValueError, match="PUF_RAW_ROSTER"):
        full.decode_full_puf_draws(**{**kwargs, "raw_draws": incomplete})
    changed_training = full.codec.decode_json(kwargs["training_state"])
    changed_training["models"][0]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="PUF_FULL_CHAIN_IDENTITY"):
        full.decode_full_puf_draws(
            **{**kwargs, "training_state": full.codec.encode_json(changed_training)}
        )
    warm = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="require"
    )
    assert warm.key == actual.key and all(record.hit for record in warm.nodes.values())
    for node in applies:
        assert payload(warm, node.id, "raw_draw") == payload(
            actual, node.id, "raw_draw"
        )
