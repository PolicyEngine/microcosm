"""Invented complete upstream control, real QRF/store/attachment, no admission.

These ordinary fixtures import maintained helpers normally. They do not extract
functions, impersonate source issuers, use genuine data, or load a tax engine.
The current diagnostic survey host does not supply this complete predictor
surface; source-qualified production integration remains an upstream duty.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_full_puf_enrichment import (
    InventedMatrix,
    InventedSource,
    _donor,
    _known,
    _recipient,
)

from microcosm.build.us_runtime import full_puf_enrichment as full
from microcosm.build.us_runtime import graph_full_puf_enrichment as placement
from microcosm.build.us_runtime import puf_support as support
from microcosm.build.us_runtime import survey_population_replay as replay
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.fit import model_input
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
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
from microcosm.graph import population as population_ops
from microcosm.graph.artifact_edges import descriptor, value_from_descriptor
from microcosm.graph.codecs import load_frame_store
from microcosm.graph.errors import NodeRejectedError
from microcosm.graph.keys import opaque_artifact_key
from microcosm.graph.population import dtype_for_token


def _copy_population(population, *, metadata=None, weights=None, mass_log=None):
    """Retain a separate comparison object, including all receiving context."""
    frame = population.frame
    return replace(
        population,
        frame=Frame(
            {entity: frame.table(entity).copy(deep=True) for entity in frame.entities},
            frame.schema,
            weights
            if weights is not None
            else {
                entity: Weights(
                    frame.weights_for(entity).values.copy(),
                    frame.weights_for(entity).kind,
                )
                for entity in frame.weighted_entities
            },
            frame.strata.copy(deep=True),
            metadata=deepcopy(dict(frame.metadata)) if metadata is None else metadata,
            mass_log=tuple(frame.mass_log) if mass_log is None else mass_log,
        ),
        owners=dict(population.owners),
        weight_kind=dict(population.weight_kind),
        design_weights={
            key: value.copy() for key, value in population.design_weights.items()
        },
    )


class InventedCompleteBoundary(KernelBase):
    """Real identity FILTER opening a version in which incumbents may rewrite."""

    ref = "test.full_puf.complete_upstream_boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        return KernelResult(
            keep=pd.Series(
                True,
                index=pd.Index(context.tables["person"].person_id, name="person_id"),
                dtype=bool,
            )
        )


def _complete_frame():
    frame = _recipient()
    frame.table("tax_unit")["health_savings_account_ald"] = np.linspace(
        701.0, 1701.0, frame.n("tax_unit"), dtype=np.float64
    )
    people = frame.table("person")
    people["self_employment_income_would_be_qualified"] = np.resize(
        np.array([False, True], dtype=bool), len(people)
    )
    people["rental_income_would_be_qualified"] = pd.array(
        np.resize(np.array([None, True], dtype=object), len(people)),
        dtype="boolean",
    )
    # Exercise preservation of unrelated columns, nullable masks, signed zero,
    # complete nested metadata, and the fixture's native ACS structural blanks.
    for entity in frame.entities:
        table = frame.table(entity)
        # The invented source declares graph-native string storage explicitly.
        for name in table.select_dtypes(include="object"):
            assert table[name].dropna().map(lambda value: isinstance(value, str)).all()
            table[name] = table[name].astype(pd.StringDtype(storage="python"))
        prefix = "" if entity == "person" else entity + "_"
        table[prefix + "unrelated_signed"] = np.resize(
            np.array([-0.0, 13.0], dtype="float64"), len(table)
        )
        table[prefix + "unrelated_nullable"] = pd.array(
            np.resize(np.array([True, None], dtype=object), len(table)), dtype="boolean"
        )
    return Frame(
        {entity: frame.table(entity).copy(deep=True) for entity in frame.entities},
        frame.schema,
        {
            # CREATE retains this separate invented design-weight vector as
            # a real Population anchor while household importance weights keep
            # the helper's assembly/clone mass contract unchanged.
            "person": Weights(
                frame.resolve_weights("person").values.copy(), WeightKind.DESIGN
            ),
            **{
                entity: Weights(
                    frame.weights_for(entity).values.copy(),
                    frame.weights_for(entity).kind,
                )
                for entity in frame.weighted_entities
            },
        },
        frame.strata.copy(deep=True),
        metadata={
            **deepcopy(dict(frame.metadata)),
            "complete_upstream_control": {
                "nested": [None, {"kept": True}],
                "zero": -0.0,
            },
            "source_admission": False,
        },
        mass_log=frame.mass_log,
    )


def _load_artifact(manifest, store, edge):
    """Read the actual typed producer descriptor and its stored payload."""
    record = manifest.node(edge.producer)
    actual_descriptor = record.typed_artifacts["outputs"][edge.artifact]
    assert actual_descriptor == descriptor(
        producer=edge.producer,
        artifact=edge.artifact,
        type_=edge.type,
        producer_key=record.key,
        capabilities=record.capabilities,
    )
    key = opaque_artifact_key(record.key, edge.artifact)
    assert record.opaque_artifacts[edge.artifact] == key
    return value_from_descriptor(store.load_bytes(key), actual_descriptor)


def _independent_expected(upstream, donor, known, nodes, artifacts):
    """Call the actual finalizer, without reusing the attachment's result builder."""
    candidate, receipt = full.finalize_full_puf(
        upstream.frame,
        donor,
        predictor_known=known,
        matrix=artifacts["matrix"].payload,
        matrix_producer_key=artifacts["matrix"].producer_key,
        raw_draws={
            target: artifacts[f"raw_{index:03d}"].payload
            for index, target in enumerate(full.TARGETS)
        },
        apply_state=artifacts["apply_state"].payload,
        training_state=artifacts["training_state"].payload,
        last_model=artifacts["last_model"].payload,
        seed=578,
    )
    mask_node, attach_node = nodes
    masks = {
        entity: support.puf_tax_detail_clone_mask(
            upstream.frame.table(entity), entity=entity
        )
        for entity in ("person", "tax_unit")
    }
    expected = population_ops.patch(
        upstream,
        mask_node,
        KernelResult(
            columns={
                (entity, placement.MASKS[entity]): pd.Series(
                    masks[entity],
                    index=pd.Index(
                        upstream.frame.table(entity)[
                            upstream.frame.schema.entity_id_column(entity)
                        ],
                        name=upstream.frame.schema.entity_id_column(entity),
                    ),
                    dtype=bool,
                )
                for entity in masks
            }
        ),
    )
    columns = {}
    for owned in attach_node.outputs:
        table = candidate.table(owned.entity)
        mask = masks[owned.entity]
        id_column = candidate.schema.entity_id_column(owned.entity)
        columns[(owned.entity, owned.column)] = pd.Series(
            table.loc[mask, owned.column].array,
            index=pd.Index(table.loc[mask, id_column], name=id_column),
            dtype=dtype_for_token(owned.dtype),
        )
    return population_ops.patch(
        expected, attach_node, KernelResult(columns=columns)
    ), receipt


@pytest.fixture(scope="module")
def attached_full65(tmp_path_factory):
    root = tmp_path_factory.mktemp("full65_attachment")
    frame, donor = _complete_frame(), _donor()
    known = _known(frame)
    prepared = full.prepare_full_puf_inputs(frame, donor, predictor_known=known)
    decoded = model_input.decode_recipient_matrix(prepared.matrix)
    table = decoded.features.copy()
    table.insert(0, "tax_unit_id", decoded.entity_ids)
    matrix_frame = Frame(
        {
            "tax_unit": table,
            "person": pd.DataFrame(
                {
                    "person_id": decoded.entity_ids,
                    "person_tax_unit_id": decoded.entity_ids,
                }
            ),
        },
        EntitySchema(group_entities=("tax_unit",)),
        {"tax_unit": Weights(np.ones(len(table), dtype="float64"), WeightKind.DESIGN)},
    )
    retained_store = ContentStore(
        root / "retained", codecs={"frame-store": load_frame_store}
    )
    sources, source_nodes = {}, []
    for name, source_frame in (
        ("survey", frame),
        ("donor", prepared.donor_frame),
        ("matrix_input", matrix_frame),
    ):
        source_name = "invented." + name
        sources[source_name] = retained_store.put_frame(
            full.codec.sha(("ordinary-full65-attachment:" + name).encode()),
            source_frame,
        )
        outputs = (
            frame_column_declarations(source_frame)
            if name == "survey"
            else tuple(
                Owned("tax_unit", column, "float64")
                for column in source_frame.table("tax_unit")
                if column != "tax_unit_id"
            )
        )
        source_nodes.append(
            Node(
                name,
                InventedSource.ref,
                sources=(source_name,),
                structural=StructuralDelta.CREATE,
                outputs=outputs,
            )
        )
    boundary = Node(
        "complete_upstream",
        InventedCompleteBoundary.ref,
        base="survey",
        structural=StructuralDelta.FILTER,
        inputs=(Slice("person", ("age",)),),
    )
    matrix_node = Node(
        "matrix",
        InventedMatrix.ref,
        population="matrix_input",
        inputs=(Slice("tax_unit", full.PREDICTORS),),
        artifact_outputs=(ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),),
    )
    base_graph = Graph(
        "us",
        tuple(SourceRef(name, "frame-store") for name in sources),
        (*source_nodes, boundary, matrix_node),
    )
    kernels = KernelRegistry()
    for kernel in (
        InventedSource(),
        InventedCompleteBoundary(),
        InventedMatrix(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        kernels.register(kernel)
    observed = {}
    retained_manifest = run_graph(
        compile_graph(base_graph),
        sources=sources,
        store=retained_store,
        kernels=kernels,
        _population_observer=lambda name, value: observed.__setitem__(name, value),
    )
    upstream = observed[boundary.id]
    independent_upstream = _copy_population(upstream)
    matrix = _load_artifact(
        retained_manifest,
        retained_store,
        ArtifactInput(
            "matrix", matrix_node.id, "matrix", model_input.RECIPIENT_MATRIX_TYPE
        ),
    )
    assert matrix.payload == prepared.matrix
    fits, applies = full.full_puf_train_apply_nodes(
        donor_population="donor",
        recipient_population=boundary.id,
        matrix_producer=matrix_node.id,
        seed=578,
        n_estimators=2,
        zero_atol=0,
    )
    retention_arguments = dict(
        population=upstream,
        expected_population=independent_upstream,
        population_node=boundary,
        input_owners=dict(upstream.owners),
        donor=donor,
        predictor_known=known,
        matrix=matrix,
        fit_nodes=fits,
        apply_nodes=applies,
    )
    binding = placement.retain_full_puf_attachment(**retention_arguments)
    attachment_nodes = placement.full_puf_attachment_nodes(binding)
    kernels.register(placement.FullPufMaskKernel(binding))
    kernels.register(placement.FullPufAttachKernel(binding))
    compiled = compile_graph(
        replace(
            base_graph, nodes=(*base_graph.nodes, *fits, *applies, *attachment_nodes)
        )
    )
    store = ContentStore(root / "application", codecs={"frame-store": load_frame_store})
    results = []
    for resume in ("auto", "require"):
        observed = {}
        manifest = run_graph(
            compiled,
            sources=sources,
            store=store,
            kernels=kernels,
            resume=resume,
            _population_observer=lambda name, value, target=observed: (
                target.__setitem__(name, value)
            ),
        )
        assert all(
            record.hit is (resume == "require") for record in manifest.nodes.values()
        )
        artifacts, producer_keys = placement.load_full_puf_attachment_artifacts(
            binding, compiled=compiled, manifest=manifest, store=store
        )
        assert artifacts == {
            edge.name: _load_artifact(manifest, store, edge)
            for edge in attachment_nodes[-1].artifact_inputs
        }
        actual_upstream = observed[boundary.id]
        actual = observed[attachment_nodes[-1].id]
        evidence = placement.verify_materialized_full_puf_attachment(
            binding,
            upstream_population=actual_upstream,
            population=actual,
            artifacts=artifacts,
            producer_keys=producer_keys,
        )
        expected, finalizer_receipt = _independent_expected(
            independent_upstream, donor, known, attachment_nodes, artifacts
        )
        replay.same_replayed_population(expected, actual)
        # Also exercise actual Frame storage, beyond warm column-artifact replay.
        frame_key = full.codec.sha(
            ("materialized:" + manifest.node(attachment_nodes[-1].id).key).encode()
        )
        store.put_frame(frame_key, actual.frame)
        replay.same_replayed_frame(expected.frame, store.load_frame(frame_key))
        results.append(
            SimpleNamespace(
                manifest=manifest,
                upstream=_copy_population(actual_upstream),
                population=_copy_population(actual),
                artifacts=artifacts,
                producer_keys=producer_keys,
                evidence=evidence,
                finalizer_receipt=finalizer_receipt,
            )
        )
    replay.same_replayed_population(results[0].population, results[1].population)
    assert results[0].artifacts == results[1].artifacts
    assert results[0].manifest.key == results[1].manifest.key
    return SimpleNamespace(
        binding=binding,
        retention_arguments=retention_arguments,
        nodes=attachment_nodes,
        fits=fits,
        applies=applies,
        results=results,
        compiled=compiled,
        sources=sources,
        store=store,
        kernels=kernels,
    )


def _verify(case, result, **changes):
    return placement.verify_materialized_full_puf_attachment(
        changes.pop("binding", case.binding),
        **{
            "upstream_population": result.upstream,
            "population": result.population,
            "artifacts": result.artifacts,
            "producer_keys": result.producer_keys,
            **changes,
        },
    )


def test_full65_cold_and_required_replay_preserve_complete_population(attached_full65):
    case = attached_full65
    mask_node, attach_node = case.nodes
    assert len(case.fits) == len(case.applies) == 65
    hsa = next(
        o
        for o in attach_node.outputs
        if (o.entity, o.column) == ("tax_unit", "health_savings_account_ald")
    )
    assert hsa.rewrite and hsa.dtype == "float64"
    for result in case.results:
        table = result.population.frame.table("tax_unit")
        original = case.binding.expected_population.frame.table("tax_unit")
        native = ~support.puf_tax_detail_clone_mask(original, entity="tax_unit")
        pd.testing.assert_series_equal(
            table.loc[native, "health_savings_account_ald"],
            original.loc[native, "health_savings_account_ald"],
            check_exact=True,
        )
    assert len(full.PERSON_OUTPUTS) == 56 and len(full.TAX_UNIT_OUTPUTS) == 9
    assert "prior_year_wages" not in full.TARGETS
    assert "employment_income_last_year" not in full.TARGETS
    assert tuple(owned.column for owned in attach_node.outputs) == full.TARGETS
    assert tuple(
        (owned.entity, owned.column, owned.dtype) for owned in mask_node.outputs
    ) == (
        ("person", placement.PERSON_MASK, "bool"),
        ("tax_unit", placement.TAX_UNIT_MASK, "bool"),
    )
    for result in case.results:
        assert result.evidence["source_admission_issued"] is False
        assert result.evidence["release_eligible"] is False
        assert result.finalizer_receipt["target_order"] == list(full.TARGETS)
        assert result.finalizer_receipt["tail_bounds"]
        frame, upstream = result.population.frame, result.upstream.frame
        assert result.population.mass_ledger == result.upstream.mass_ledger
        assert frame.mass_log == upstream.mass_log
        expected_new = {
            ("person", placement.PERSON_MASK),
            ("tax_unit", placement.TAX_UNIT_MASK),
            *(("person", name) for name in full.PERSON_OUTPUTS),
            *(("tax_unit", name) for name in full.TAX_UNIT_OUTPUTS),
        }
        for cell, owner in result.upstream.owners.items():
            if cell not in expected_new:
                assert result.population.owners[cell] == owner
        for owned in attach_node.outputs:
            table, before = frame.table(owned.entity), upstream.table(owned.entity)
            mask = support.puf_tax_detail_clone_mask(before, entity=owned.entity)
            assert owned.rows == placement.MASKS[owned.entity]
            assert table[owned.column].dtype == dtype_for_token(owned.dtype)
            assert table.loc[mask, owned.column].notna().all()
            assert (
                result.population.owners[(owned.entity, owned.column)] == attach_node.id
            )
            if owned.column in before:
                assert owned.rewrite is True
                assert population_ops.storage_equal(
                    before[owned.column], table[owned.column], ~mask
                )
            else:
                assert owned.rewrite is False
                assert table.loc[~mask, owned.column].isna().all()
            np.testing.assert_array_equal(table[owned.rows], mask)
        assert frame.table("person").is_full_time_college_student.dtype == np.dtype(
            "bool"
        )
        assert frame.table(
            "person"
        ).self_employment_income_would_be_qualified.dtype == np.dtype("bool")
        assert (
            frame.table("person").rental_income_would_be_qualified.dtype
            == pd.BooleanDtype()
        )
        assert frame.table("person").business_is_sstb.dtype == pd.BooleanDtype()
        people = frame.table("person")
        children = people[placement.PERSON_MASK] & people.age.lt(15)
        assert people.loc[children, "employment_income_before_lsr"].eq(0).all()
        tax_mask = frame.table("tax_unit")[placement.TAX_UNIT_MASK].to_numpy()
        assert np.any(upstream.resolve_weights("tax_unit").values[tax_mask] == 0)
        decoded = model_input.decode_recipient_matrix(
            result.artifacts["matrix"].payload
        )
        assert len(decoded.features) == int(tax_mask.sum())


@pytest.mark.parametrize(
    "mutation",
    (
        "raw_values",
        "matrix_values",
        "model_bytes",
        "training_history",
        "apply_history",
        "raw_roster",
        "raw_key",
        "wrong_type",
        "apply_sibling",
        "model_sibling",
        "producer_key",
    ),
)
def test_actual_artifact_value_identity_and_sibling_mutations_are_refused(
    attached_full65, mutation
):
    case, result = attached_full65, attached_full65.results[-1]
    artifacts = dict(result.artifacts)
    producer_keys = dict(result.producer_keys)
    matrix = model_input.decode_recipient_matrix(artifacts["matrix"].payload)
    if mutation == "raw_values":
        raw = full.codec.read_raw_target(
            artifacts["raw_000"].payload,
            target=full.TARGETS[0],
            index=matrix.features.index,
        ).copy()
        raw[0] += 1.0
        artifacts["raw_000"] = replace(
            artifacts["raw_000"],
            payload=full.codec.encode_raw_target(
                raw, target=full.TARGETS[0], index=matrix.features.index
            ),
        )
    elif mutation == "matrix_values":
        features = matrix.features.copy(deep=True)
        features.iloc[0, 2] += 1.0
        artifacts["matrix"] = replace(
            artifacts["matrix"],
            payload=model_input.encode_recipient_matrix(
                features, entity="tax_unit", entity_ids=matrix.entity_ids
            ),
        )
    elif mutation == "model_bytes":
        artifacts["last_model"] = replace(
            artifacts["last_model"],
            payload=artifacts["last_model"].payload + b"changed",
        )
    elif mutation == "training_history":
        changed = full.codec.decode_json(artifacts["training_state"].payload)
        changed["models"][0]["sha256"] = "e" * 64
        artifacts["training_state"] = replace(
            artifacts["training_state"], payload=full.codec.encode_json(changed)
        )
    elif mutation == "apply_history":
        changed = full.codec.decode_json(artifacts["apply_state"].payload)
        changed["application"]["raw_targets"][0]["sha256"] = "e" * 64
        artifacts["apply_state"] = replace(
            artifacts["apply_state"], payload=full.codec.encode_json(changed)
        )
    elif mutation == "raw_roster":
        artifacts.pop("raw_064")
    elif mutation == "raw_key":
        artifacts["raw_000"] = replace(
            artifacts["raw_000"], key=artifacts["raw_001"].key
        )
    elif mutation == "wrong_type":
        artifacts["last_model"] = replace(
            artifacts["last_model"], type=ArtifactType("test.unrelated", 1)
        )
    elif mutation in ("apply_sibling", "model_sibling"):
        alias = "apply_state" if mutation == "apply_sibling" else "training_state"
        artifacts[alias] = replace(
            artifacts[alias],
            producer_key="f" * 64,
            key=opaque_artifact_key("f" * 64, alias),
        )
        # Even a matching altered node-key map cannot pair this with the actual
        # last raw draw/model sibling, whose actual producer key still differs.
        node_id = (
            case.applies[-1].id if mutation == "apply_sibling" else case.fits[-1].id
        )
        producer_keys[node_id] = "f" * 64
    elif mutation == "producer_key":
        producer_keys[case.applies[0].id] = "e" * 64
    expected_refusal = (
        r"^Legacy QRF artifact content digest mismatch\.$"
        if mutation == "model_bytes"
        else "FULL_PUF_|SURVEY_POPULATION_REPLAY_|PUF_"
    )
    with pytest.raises(ValueError, match=expected_refusal):
        _verify(case, result, artifacts=artifacts, producer_keys=producer_keys)


@pytest.mark.parametrize(
    "mutation",
    (
        "metadata",
        "weights",
        "design_anchor",
        "output_owner",
        "unrelated_owner",
        "strata",
        "membership",
        "ids",
        "unrelated_value",
        "native_output",
        "clone_output",
        "person_mask",
        "tax_unit_mask",
        "mass_ledger",
        "frame_mass_log",
    ),
)
def test_complete_materialized_population_mutations_are_refused(
    attached_full65, mutation
):
    case, result = attached_full65, attached_full65.results[-1]
    changed = _copy_population(result.population)
    people = changed.frame.table("person")
    if mutation == "metadata":
        changed = _copy_population(
            changed,
            metadata={
                **dict(changed.frame.metadata),
                "unexpected": {"nested": [False]},
            },
        )
    elif mutation == "weights":
        weights = {
            entity: Weights(
                changed.frame.weights_for(entity).values.copy(),
                changed.frame.weights_for(entity).kind,
            )
            for entity in changed.frame.weighted_entities
        }
        entity = changed.frame.weighted_entities[0]
        values = weights[entity].values.copy()
        values[0] += 0.25
        weights[entity] = Weights(values, weights[entity].kind)
        changed = _copy_population(changed, weights=weights)
    elif mutation == "design_anchor":
        design = {
            entity: values.copy() for entity, values in changed.design_weights.items()
        }
        assert design
        design[next(iter(design))][0] += 0.25
        changed = replace(changed, design_weights=design)
    elif mutation in ("output_owner", "unrelated_owner"):
        column = (
            full.PERSON_OUTPUTS[0] if mutation == "output_owner" else "unrelated_signed"
        )
        changed = replace(
            changed, owners={**changed.owners, ("person", column): "different.owner"}
        )
    elif mutation == "strata":
        changed.frame.strata.name = "changed_strata_name"
    elif mutation == "membership":
        people.loc[people.index[0], "person_tax_unit_id"] = (
            people.person_tax_unit_id.iloc[-1]
        )
    elif mutation == "ids":
        people.loc[people.index[0], "person_id"] += 1_000_000
    elif mutation == "unrelated_value":
        people.loc[people.index[0], "unrelated_signed"] = (
            0.0  # Signed-zero bits changed.
        )
    elif mutation in ("native_output", "clone_output"):
        mask = people[placement.PERSON_MASK].to_numpy()
        selected = ~mask if mutation == "native_output" else mask
        people.loc[people.index[selected][0], "employment_income_before_lsr"] += 1.0
    elif mutation in ("person_mask", "tax_unit_mask"):
        entity = "person" if mutation == "person_mask" else "tax_unit"
        table = changed.frame.table(entity)
        table.loc[table.index[0], placement.MASKS[entity]] = ~table.loc[
            table.index[0], placement.MASKS[entity]
        ]
    elif mutation == "mass_ledger":
        assert changed.mass_ledger
        changed = replace(changed, mass_ledger=changed.mass_ledger[:-1])
    elif mutation == "frame_mass_log":
        assert changed.frame.mass_log
        changed = _copy_population(changed, mass_log=changed.frame.mass_log[:-1])
    with pytest.raises(ValueError, match="FULL_PUF_|SURVEY_POPULATION_REPLAY_|PUF_"):
        _verify(case, result, population=changed)


@pytest.mark.parametrize(
    "mutation",
    ("unrelated_value", "column_missing", "column_order", "owner", "metadata"),
)
def test_upstream_replay_checks_full_state_beyond_matrix_values(
    attached_full65, mutation
):
    case, result = attached_full65, attached_full65.results[-1]
    changed = _copy_population(result.upstream)
    people = changed.frame.table("person")
    if mutation == "unrelated_value":
        people.loc[people.index[0], "unrelated_signed"] = 9.0
    elif mutation == "column_missing":
        people.drop(columns="unrelated_signed", inplace=True)
    elif mutation == "column_order":
        values = people.pop("unrelated_signed")
        people.insert(0, "unrelated_signed", values)
    elif mutation == "owner":
        changed = replace(
            changed,
            owners={**changed.owners, ("person", "age"): "different.upstream.owner"},
        )
    elif mutation == "metadata":
        changed = _copy_population(changed, metadata={"matrix_values_unchanged": True})
    # The actual model matrix is unaffected by every mutation in this group.
    assert (
        full.prepare_full_puf_inputs(
            changed.frame,
            case.retention_arguments["donor"],
            predictor_known=case.retention_arguments["predictor_known"],
        ).matrix
        == result.artifacts["matrix"].payload
    )
    with pytest.raises(ValueError, match="FULL_PUF_|SURVEY_POPULATION_REPLAY_|PUF_"):
        _verify(case, result, upstream_population=changed)


@pytest.mark.parametrize(
    "mutation",
    (
        "donor",
        "knownness",
        "population",
        "expected_population",
        "expected_null_backing",
        "owners",
    ),
)
def test_retained_live_inputs_cannot_change_after_binding(attached_full65, mutation):
    case, result = attached_full65, attached_full65.results[-1]
    arguments = {
        **case.retention_arguments,
        "population": _copy_population(case.retention_arguments["population"]),
        "expected_population": _copy_population(
            case.retention_arguments["expected_population"]
        ),
        "donor": case.retention_arguments["donor"].copy(deep=True),
        "predictor_known": case.retention_arguments["predictor_known"].copy(deep=True),
        "input_owners": dict(case.retention_arguments["input_owners"]),
    }
    binding = placement.retain_full_puf_attachment(**arguments)
    if mutation == "donor":
        arguments["donor"].loc[arguments["donor"].index[0], "casualty_loss"] += 1.0
    elif mutation == "knownness":
        arguments["predictor_known"].iloc[0, 0] = False
    elif mutation == "population":
        table = arguments["population"].frame.table("person")
        table.loc[table.index[0], "unrelated_signed"] += 1.0
    elif mutation == "expected_population":
        table = arguments["expected_population"].frame.table("person")
        table.loc[table.index[0], "unrelated_signed"] += 1.0
    elif mutation == "expected_null_backing":
        values = (
            arguments["expected_population"]
            .frame.table("person")["unrelated_nullable"]
            .array
        )
        position = np.flatnonzero(values._mask)[0]
        values._data[position] = ~values._data[position]
    elif mutation == "owners":
        binding = replace(
            binding,
            input_owners={**binding.input_owners, ("person", "age"): "different.owner"},
        )
    with pytest.raises(ValueError, match="FULL_PUF_|SURVEY_POPULATION_REPLAY_|PUF_"):
        _verify(case, result, binding=binding)


@pytest.mark.parametrize(
    "mutation",
    (
        "short_chain",
        "fit_seed",
        "apply_matrix",
        "create_population",
        "incomplete_owner_map",
    ),
)
def test_boundary_requires_complete_chain_and_explicit_population_owner_binding(
    attached_full65, mutation
):
    arguments = dict(attached_full65.retention_arguments)
    if mutation == "short_chain":
        arguments["fit_nodes"] = arguments["fit_nodes"][:-1]
        arguments["apply_nodes"] = arguments["apply_nodes"][:-1]
    elif mutation == "fit_seed":
        fits = arguments["fit_nodes"]
        arguments["fit_nodes"] = (
            replace(fits[0], params={**fits[0].params, "seed": 579}),
            *fits[1:],
        )
    elif mutation == "apply_matrix":
        applies = arguments["apply_nodes"]
        edges = tuple(
            replace(edge, producer="different.matrix")
            if edge.name == "matrix"
            else edge
            for edge in applies[0].artifact_inputs
        )
        arguments["apply_nodes"] = (
            replace(applies[0], artifact_inputs=edges),
            *applies[1:],
        )
    elif mutation == "create_population":
        arguments["population_node"] = Node(
            "complete_upstream",
            InventedSource.ref,
            structural=StructuralDelta.CREATE,
            sources=("invented.survey",),
            outputs=(Owned("person", "age", "float64"),),
        )
    elif mutation == "incomplete_owner_map":
        arguments["input_owners"] = dict(arguments["input_owners"])
        arguments["input_owners"].pop(("person", "age"))
    with pytest.raises(ValueError, match="FULL_PUF_|SURVEY_POPULATION_REPLAY_|PUF_"):
        placement.retain_full_puf_attachment(**arguments)


def test_actual_graph_refuses_omitted_unrelated_attachment_input(attached_full65):
    case = attached_full65
    attach = case.nodes[-1]
    changed = replace(
        attach,
        inputs=tuple(
            replace(
                slice_,
                columns=tuple(
                    name for name in slice_.columns if name != "unrelated_signed"
                ),
            )
            if slice_.entity == "person"
            else slice_
            for slice_ in attach.inputs
        ),
    )
    assert changed.inputs != attach.inputs
    compiled = compile_graph(
        replace(
            case.compiled.graph,
            nodes=tuple(
                changed if node.id == attach.id else node
                for node in case.compiled.graph.nodes
            ),
        )
    )
    with pytest.raises(NodeRejectedError, match="FULL_PUF_DECLARATION"):
        run_graph(
            compiled, sources=case.sources, store=case.store, kernels=case.kernels
        )


@pytest.mark.parametrize(
    "mutation",
    ("unknown_predictor", "monetary_float32", "numeric_boolean", "incomplete_matrix"),
)
def test_complete_input_knownness_and_incumbent_physical_dtypes_are_required(
    attached_full65, mutation
):
    case = attached_full65
    arguments = {
        **case.retention_arguments,
        "population": _copy_population(case.retention_arguments["population"]),
        "predictor_known": case.retention_arguments["predictor_known"].copy(deep=True),
    }
    frame = arguments["population"].frame
    if mutation == "unknown_predictor":
        arguments["predictor_known"].iloc[0, 0] = False
    elif mutation == "monetary_float32":
        table = frame.table("person")
        table["employment_income_before_lsr"] = (
            table.employment_income_before_lsr.astype("float32")
        )
    elif mutation == "numeric_boolean":
        table = frame.table("person")
        table["self_employment_income_would_be_qualified"] = (
            table.self_employment_income_would_be_qualified.astype("float64")
        )
    elif mutation == "incomplete_matrix":
        value = arguments["matrix"]
        decoded = model_input.decode_recipient_matrix(value.payload)
        arguments["matrix"] = replace(
            value,
            payload=model_input.encode_recipient_matrix(
                decoded.features.iloc[:, :-1],
                entity="tax_unit",
                entity_ids=decoded.entity_ids,
            ),
        )
    arguments["expected_population"] = _copy_population(arguments["population"])
    with pytest.raises(ValueError, match="FULL_PUF_|SURVEY_POPULATION_REPLAY_|PUF_"):
        placement.retain_full_puf_attachment(**arguments)


@pytest.mark.parametrize(
    "mutation",
    (
        "kernel_ref",
        "kernel_impl",
        "producer_key",
        "producer_descriptor_type",
        "consumer_descriptor_type",
        "compiled_fit_params",
        "compiled_apply_edge",
    ),
)
def test_actual_manifest_and_compiled_producer_contract_mutations_are_refused(
    attached_full65, mutation
):
    case, result = attached_full65, attached_full65.results[-1]
    manifest, compiled = result.manifest, case.compiled
    records = dict(manifest.nodes)
    node_id = case.fits[-1].id
    record = records[node_id]
    if mutation == "kernel_ref":
        records[node_id] = replace(record, kernel_ref="test.different.fit@1")
    elif mutation == "kernel_impl":
        records[node_id] = replace(record, kernel_impl_hash="e" * 64)
    elif mutation == "producer_key":
        records[node_id] = replace(record, key="e" * 64)
    elif mutation in ("producer_descriptor_type", "consumer_descriptor_type"):
        if mutation == "consumer_descriptor_type":
            node_id = case.nodes[-1].id
            record = records[node_id]
        typed = {
            direction: {
                name: {**value, "type": dict(value["type"])}
                for name, value in entries.items()
            }
            for direction, entries in record.typed_artifacts.items()
        }
        direction, name = (
            ("outputs", "model")
            if mutation == "producer_descriptor_type"
            else ("inputs", "last_model")
        )
        typed[direction][name]["type"]["name"] = "test.different.model"
        records[node_id] = replace(record, typed_artifacts=typed)
    elif mutation == "compiled_fit_params":
        fit = case.fits[-1]
        changed = replace(fit, params={**fit.params, "n_estimators": 3})
        compiled = compile_graph(
            replace(
                compiled.graph,
                nodes=tuple(
                    changed if node.id == fit.id else node
                    for node in compiled.graph.nodes
                ),
            )
        )
    elif mutation == "compiled_apply_edge":
        apply = case.applies[-1]
        changed = replace(
            apply,
            artifact_inputs=tuple(
                replace(edge, producer=case.applies[1].id)
                if edge.name == "prior_000"
                else edge
                for edge in apply.artifact_inputs
            ),
        )
        compiled = compile_graph(
            replace(
                compiled.graph,
                nodes=tuple(
                    changed if node.id == apply.id else node
                    for node in compiled.graph.nodes
                ),
            )
        )
    if mutation in (
        "producer_key",
        "producer_descriptor_type",
        "consumer_descriptor_type",
    ):
        # Make both ends of typed ancestry internally consistent. Construction
        # must succeed; the owned loader then rejects the false graph contract.
        producer_id = case.fits[-1].id
        producer = records[producer_id]
        changed_type = ArtifactType("test.different.model", 1)
        outputs = {
            name: descriptor(
                producer=producer_id,
                artifact=name,
                type_=(
                    changed_type
                    if name == "model" and mutation != "producer_key"
                    else ArtifactType(
                        value["type"]["name"], value["type"]["schema_version"]
                    )
                ),
                producer_key=producer.key,
                capabilities=producer.capabilities,
            )
            for name, value in producer.typed_artifacts["outputs"].items()
        }
        records[producer_id] = replace(
            producer,
            typed_artifacts={**producer.typed_artifacts, "outputs": outputs},
            opaque_artifacts={name: value["key"] for name, value in outputs.items()},
        )
        for consumer_id, consumer in tuple(records.items()):
            if not consumer.typed_artifacts:
                continue
            inputs = {
                name: outputs[value["artifact"]]
                if value["producer"] == producer_id
                else value
                for name, value in consumer.typed_artifacts["inputs"].items()
            }
            records[consumer_id] = replace(
                consumer, typed_artifacts={**consumer.typed_artifacts, "inputs": inputs}
            )
    manifest = replace(manifest, nodes=records)
    with pytest.raises(ValueError, match="FULL_PUF_"):
        placement.load_full_puf_attachment_artifacts(
            case.binding, compiled=compiled, manifest=manifest, store=case.store
        )


def test_person_and_tax_unit_masks_must_describe_the_same_memberships(attached_full65):
    arguments = dict(attached_full65.retention_arguments)
    population = _copy_population(arguments["population"])
    people, units = population.frame.table("person"), population.frame.table("tax_unit")
    person_mask = support.puf_tax_detail_clone_mask(people, entity="person")
    unit_mask = support.puf_tax_detail_clone_mask(units, entity="tax_unit")
    people.loc[people.index[person_mask][0], "person_tax_unit_id"] = units.loc[
        ~unit_mask, "tax_unit_id"
    ].iloc[0]
    arguments.update(
        population=population,
        expected_population=_copy_population(population),
    )
    # Both retained inputs agree on this malformed candidate; the independent
    # mask-membership invariant, rather than a changed-state seal, must reject it.
    with pytest.raises(ValueError, match="FULL_PUF_MASK_MEMBERSHIP"):
        placement.retain_full_puf_attachment(**arguments)


def test_rebuilt_binding_from_fresh_frame_store_requires_full_cached_replay(
    attached_full65, tmp_path
):
    """Nullable physical mutation seals cannot become cross-session graph keys."""
    case = attached_full65
    original = case.retention_arguments
    cold_expected = _copy_population(original["expected_population"])
    # Exercise the real codec's difference between masked storage and value.
    hidden = cold_expected.frame.table("person")["unrelated_nullable"].array
    nulls = hidden._mask
    assert nulls.any()
    hidden._data[nulls] = True
    assert hidden._data[nulls].all()
    cold_binding = placement.retain_full_puf_attachment(
        **{**original, "expected_population": cold_expected}
    )
    cold_nodes = placement.full_puf_attachment_nodes(cold_binding)
    assert cold_nodes == case.nodes

    retained_path = tmp_path / "new-session-upstream"
    writer = ContentStore(retained_path)
    frame_key = full.codec.sha(b"rebuilt-full65-upstream")
    writer.put_frame(frame_key, cold_expected.frame)
    # A distinct store object reads actual codec output; no in-memory reuse.
    reader = ContentStore(retained_path)
    restored = reader.load_frame(frame_key)
    assert not restored.person["unrelated_nullable"].array._data[nulls].any()
    replay.same_replayed_frame(cold_expected.frame, restored)
    warm_expected = replace(_copy_population(cold_expected), frame=restored)
    warm_upstream = _copy_population(case.results[-1].upstream)
    warm_binding = placement.retain_full_puf_attachment(
        **{
            **original,
            "population": warm_upstream,
            "expected_population": warm_expected,
            "input_owners": dict(warm_upstream.owners),
        }
    )
    assert cold_binding.expected_stamp != warm_binding.expected_stamp
    warm_nodes = placement.full_puf_attachment_nodes(warm_binding)
    assert warm_nodes == cold_nodes
    assert placement._placement(warm_binding) == placement._placement(cold_binding)
    rebuilt_graph = compile_graph(
        replace(
            case.compiled.graph,
            nodes=tuple(
                {node.id: node for node in warm_nodes}.get(node.id, node)
                for node in case.compiled.graph.nodes
            ),
        )
    )
    kernels = KernelRegistry()
    for kernel in (
        InventedSource(),
        InventedCompleteBoundary(),
        InventedMatrix(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
        placement.FullPufMaskKernel(warm_binding),
        placement.FullPufAttachKernel(warm_binding),
    ):
        kernels.register(kernel)
    observed = {}
    manifest = run_graph(
        rebuilt_graph,
        sources=case.sources,
        store=case.store,
        kernels=kernels,
        resume="require",
        _population_observer=lambda name, value: observed.__setitem__(name, value),
    )
    assert all(record.hit for record in manifest.nodes.values())
    assert manifest.key == case.results[-1].manifest.key
    artifacts, producer_keys = placement.load_full_puf_attachment_artifacts(
        warm_binding, compiled=rebuilt_graph, manifest=manifest, store=case.store
    )
    placement.verify_materialized_full_puf_attachment(
        warm_binding,
        upstream_population=observed[warm_binding.population_node.id],
        population=observed[warm_nodes[-1].id],
        artifacts=artifacts,
        producer_keys=producer_keys,
    )
    replay.same_replayed_population(
        case.results[-1].population, observed[warm_nodes[-1].id]
    )
