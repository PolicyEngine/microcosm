"""Ordinary canonical-column controls; no genuine donor/source authority mocks."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import full_puf_enrichment as full
from microcosm.build.us_runtime import puf_support as support
from microcosm.build.us_runtime.acs_income_universe import (
    apply_acs_pums_earnings_universe_zeros,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.fit import model_input
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import load_frame_store
from microcosm.graph.keys import opaque_artifact_key


def _donor_columns():
    links = np.repeat(np.arange(1001, 1005, dtype=np.int64), 2)
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 109, dtype=np.int64),
            "person_tax_unit_id": links,
        }
    )
    for index, target in enumerate(full.PERSON_OUTPUTS):
        person[target] = (
            np.arange(8) % 2 == 1
            if target in support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
            else (np.arange(8, dtype=np.float64) + 1) * (index + 1)
        )
    person["short_term_capital_gains"] = [-10.0, 0.0, 5.0, 10.0, -3.0, 0.0, 12.0, 20.0]
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": np.arange(1001, 1005, dtype=np.int64),
            "weight": [1.0, 0.0, 2.0, 3.0],
            "filing_status_code": [1.0, 2.0, 3.0, 4.0],
        }
    )
    for target in full.TAX_UNIT_OUTPUTS:
        tax_unit[target] = (
            [0.0, 2010.0, 2020.0, 2024.0]
            if target in support._PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS
            else [1.0, 2.0, 3.0, 4.0]
        )
    pk = pd.DataFrame(True, index=person.index, columns=full.PERSON_OUTPUTS)
    tk = pd.DataFrame(
        True,
        index=tax_unit.index,
        columns=("weight", "filing_status_code", *full.TAX_UNIT_OUTPUTS),
    )
    return person, tax_unit, pk, tk


def _donor():
    p, t, pk, tk = _donor_columns()
    return full.canonical_full_puf_donor(p, t, person_known=pk, tax_unit_known=tk)


def _recipient():
    ids = np.arange(1, 5, dtype=np.int64)
    links = np.repeat(ids, 2)
    person = pd.DataFrame({"person_id": np.arange(1, 9, dtype=np.int64)})
    for entity in US_SCHEMA.group_entities:
        person[US_SCHEMA.membership_column(entity)] = links
    person["age"] = [14.0, 45.0] * 4
    for name, amount in (
        ("employment_income_before_lsr", 100.0),
        ("self_employment_income_before_lsr", 20.0),
        ("taxable_interest_income", 5.0),
        ("qualified_dividend_income", 2.0),
        ("non_qualified_dividend_income", 3.0),
        ("short_term_capital_gains", 10.0),
        ("long_term_capital_gains_before_response", 12.0),
    ):
        person[name] = [0.0, amount] * 4
    for name in support.PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS:
        person[name] = [0.0, 10.0] * 4
    person["is_full_time_college_student"] = [False, True] * 4
    tables = {
        entity: pd.DataFrame({US_SCHEMA.entity_id_column(entity): ids})
        for entity in US_SCHEMA.group_entities
    }
    tables["tax_unit"]["filing_status_input"] = [
        "SINGLE",
        "JOINT",
        "SEPARATE",
        "HEAD_OF_HOUSEHOLD",
    ]
    tables["person"] = person
    native = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([1.0, 0.0, 2.0, 3.0]), WeightKind.DESIGN)},
        metadata={"scope": "invented_full_puf_contract", "source_admission": False},
    )
    asec = native.select(person.person_household_id.le(2))
    acs = native.select(person.person_household_id.gt(2))
    acs_person = acs.table("person")
    child = acs_person.age.lt(15)
    for mapped, raw in (
        ("employment_income_before_lsr", "WAGP"),
        ("self_employment_income_before_lsr", "SEMP"),
    ):
        acs_person[raw] = acs_person[mapped]
        acs_person.loc[child, [mapped, raw]] = np.nan
    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    cloned = support.clone_us_frame_for_puf_support(assembled)
    application = apply_acs_pums_earnings_universe_zeros(
        cloned, boundary="invented full65 recipient ACS materialization"
    )
    # This is the maintained source-universe operator, with its actual receipt;
    # the mapped child zeros are produced here and the raw blanks survive.
    assert application.receipt["structurally_absent_person_rows"] == 4
    people = application.frame.table("person")
    acs_child = people[support.support_channel_column("person")].eq(
        "acs"
    ) & people.age.lt(15)
    assert people.loc[acs_child, ["WAGP", "SEMP"]].isna().all().all()
    return application.frame


def _known(frame):
    table = frame.table("tax_unit")
    mask = support.puf_tax_detail_clone_mask(table, entity="tax_unit")
    return pd.DataFrame(
        True,
        columns=full.PREDICTORS,
        index=pd.Index(table.loc[mask, "tax_unit_id"].to_numpy(), name="tax_unit_id"),
    )


def test_complete_canonical_reduction_retains_zero_weights_signed_values_and_mortgage():
    p, t, pk, tk = _donor_columns()
    before = p.copy(deep=True)
    donor = full.canonical_full_puf_donor(p, t, person_known=pk, tax_unit_known=tk)
    assert len(full.TARGETS) == 65 and len(full.PERSON_OUTPUTS) == 56
    assert "prior_year_wages" not in full.TARGETS
    assert tuple(donor) == (*full.PREDICTORS, *full.TARGETS, "weight")
    assert donor.weight.tolist() == [1.0, 0.0, 2.0, 3.0]
    assert donor.short_term_capital_gains.tolist() == [-10.0, 15.0, -3.0, 32.0]
    assert donor.business_is_sstb.tolist() == [1.0] * 4
    # Canonical mortgage has already been interpreted upstream: no E19200 split.
    np.testing.assert_array_equal(
        donor.home_mortgage_interest,
        p.home_mortgage_interest.groupby(p.person_tax_unit_id).sum(),
    )
    pd.testing.assert_frame_equal(p, before)


@pytest.mark.parametrize(
    "value",
    [None, np.nan, np.inf, "0", True, 1 + 2j, pd.Timestamp("2024-01-01"), 2**53 + 1],
)
def test_rejects_bad_member_before_groupby_can_turn_it_into_a_valid_total(value):
    p, t, pk, tk = _donor_columns()
    p["taxable_interest_income"] = p.taxable_interest_income.astype(object)
    p.loc[0, "taxable_interest_income"] = value
    with pytest.raises(
        ValueError, match="PUF_(UNKNOWN|NONFINITE|PHYSICAL_TYPE|FLOAT64_INTEGER_RANGE)"
    ):
        full.canonical_full_puf_donor(p, t, person_known=pk, tax_unit_known=tk)


def test_unknown_zero_and_numeric_boolean_are_not_complete_canonical_values():
    p, t, pk, tk = _donor_columns()
    p.loc[0, "taxable_interest_income"] = 0.0
    pk.loc[0, "taxable_interest_income"] = False
    with pytest.raises(ValueError, match="PUF_UNKNOWN:person.taxable_interest_income"):
        full.canonical_full_puf_donor(p, t, person_known=pk, tax_unit_known=tk)
    pk.loc[0, "taxable_interest_income"] = True
    p["business_is_sstb"] = p.business_is_sstb.astype(np.int64)
    with pytest.raises(ValueError, match="PUF_PHYSICAL_TYPE:person.business_is_sstb"):
        full.canonical_full_puf_donor(p, t, person_known=pk, tax_unit_known=tk)


def test_explicit_return_grain_for_person_destination_and_collision_refusal():
    p, t, pk, tk = _donor_columns()
    name = "self_employed_pension_contributions_desired"
    t[name] = [10.0, 20.0, 30.0, 40.0]
    tk[name] = True
    tk = tk.loc[:, ["weight", "filing_status_code", name, *full.TAX_UNIT_OUTPUTS]]
    kwargs = dict(
        person_known=pk.drop(columns=name),
        tax_unit_known=tk,
        person_targets_at_tax_unit=(name,),
    )
    with pytest.raises(ValueError, match="PUF_DONOR_GRAIN_COLLISION"):
        full.canonical_full_puf_donor(p, t, **kwargs)
    donor = full.canonical_full_puf_donor(p.drop(columns=name), t, **kwargs)
    assert donor[name].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_full_recipient_surface_keeps_zero_weight_rows_and_rejects_unknowns():
    frame, donor = _recipient(), _donor()
    known = _known(frame)
    inputs = full.prepare_full_puf_inputs(frame, donor, predictor_known=known)
    matrix = model_input.decode_recipient_matrix(inputs.matrix)
    assert len(matrix.features) == 4 and tuple(matrix.features) == full.PREDICTORS
    assert matrix.entity_ids.tolist() == known.index.tolist()
    assert inputs.recipient_universe["structurally_absent_person_rows"] == 2
    assert matrix.features[full.PREDICTORS[2]].eq(100.0).all()
    assert matrix.features[full.PREDICTORS[3]].eq(20.0).all()
    assert (frame.weights_for("household").values == 0).sum() == 2
    assert (
        inputs.recipient_universe["person_output_allocation"][
            "out_of_universe_person_rows"
        ]
        == 4
    )
    known.iloc[0, 2] = False
    with pytest.raises(ValueError, match="PUF_UNKNOWN:recipient_predictor"):
        full.prepare_full_puf_inputs(frame, donor, predictor_known=known)
    known.iloc[0, 2] = True
    person = frame.table("person")
    person.loc[person.index[-1], "taxable_interest_income"] = np.nan
    with pytest.raises(ValueError, match="missing values before coercion"):
        full.prepare_full_puf_inputs(frame, donor, predictor_known=known)


def test_return_only_donor_does_not_invent_persons():
    expected = _donor()
    _, returns, _, _ = _donor_columns()
    columns = (
        "weight",
        "filing_status_code",
        "tax_unit_person_count",
        *full.PERSON_OUTPUTS,
        *full.TAX_UNIT_OUTPUTS,
    )
    returns["tax_unit_person_count"] = expected[full.PREDICTORS[1]]
    for name in full.PERSON_OUTPUTS:
        returns[name] = expected[name]
    known = pd.DataFrame(True, index=returns.index, columns=columns)
    actual = full.canonical_full_puf_donor(
        None,
        returns,
        person_known=None,
        tax_unit_known=known,
        person_targets_at_tax_unit=full.PERSON_OUTPUTS,
    )
    pd.testing.assert_frame_equal(actual, expected)


class InventedSource(KernelBase):
    ref = "test.full_puf.frame_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_frame_store(context.sources[context.node.sources[0]])
        )


class InventedMatrix(KernelBase):
    ref = "test.full_puf.matrix@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def run(self, context):
        table = context.tables["tax_unit"]
        ids = table.tax_unit_id.to_numpy()
        features = table.loc[:, list(full.PREDICTORS)].copy()
        features.index = pd.Index(ids, name="tax_unit_id")
        return KernelResult(
            artifacts={
                "matrix": model_input.encode_recipient_matrix(
                    features, entity="tax_unit", entity_ids=ids
                )
            }
        )


def test_real_full_65_target_graph_draw_chain_finalization_and_replay(tmp_path):
    frame, donor = _recipient(), _donor()
    known = _known(frame)
    prepared = full.prepare_full_puf_inputs(frame, donor, predictor_known=known)
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
    )
    assert len(fits) == len(applies) == 65
    assert [
        e.name for e in applies[-1].artifact_inputs if e.name.startswith("prior_")
    ] == [f"prior_{i:03d}" for i in range(64)]
    nodes.append(
        Node(
            "matrix",
            InventedMatrix.ref,
            population="recipient",
            inputs=(Slice("tax_unit", full.PREDICTORS),),
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
        InventedMatrix(),
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
        for target, node in zip(full.TARGETS, applies, strict=True)
    }
    kwargs = dict(
        matrix=payload(actual, "matrix", "matrix"),
        matrix_producer_key=actual.node("matrix").key,
        raw_draws=raw,
        apply_state=payload(actual, applies[-1].id, "apply_state"),
        training_state=payload(actual, fits[-1].id, "training_state"),
        seed=578,
    )
    assert kwargs["matrix"] == prepared.matrix
    before = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    last_model = payload(actual, fits[-1].id, "model")
    result, receipt = full.finalize_full_puf(
        frame, donor, predictor_known=known, last_model=last_model, **kwargs
    )
    assert receipt["release_eligible"] is False and receipt["tail_bounds"]
    assert receipt["target_order"] == list(full.TARGETS)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(frame.table(entity), before[entity])
    np.testing.assert_array_equal(
        result.weights_for("household").values, frame.weights_for("household").values
    )
    for entity, outputs in (
        ("person", full.PERSON_OUTPUTS),
        ("tax_unit", full.TAX_UNIT_OUTPUTS),
    ):
        mask = support.puf_tax_detail_clone_mask(result.table(entity), entity=entity)
        for output in outputs:
            assert result.table(entity).loc[mask, output].notna().all()
            if output not in before[entity]:
                assert result.table(entity).loc[~mask, output].isna().all()
            else:
                np.testing.assert_array_equal(
                    result.table(entity).loc[~mask, output],
                    before[entity].loc[~mask, output],
                )
    people = result.table("person")
    child = support.puf_tax_detail_clone_mask(people, entity="person") & people.age.lt(
        15
    )
    assert people.loc[child, "employment_income_before_lsr"].eq(0).all()
    changed_raw = dict(raw)
    values = full.codec.read_raw_target(
        raw[full.TARGETS[0]], target=full.TARGETS[0], index=matrix.features.index
    ).copy()
    values[0] += 1.0
    changed_raw[full.TARGETS[0]] = full.codec.encode_raw_target(
        values, target=full.TARGETS[0], index=matrix.features.index
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
    incomplete.pop(full.TARGETS[-1])
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
