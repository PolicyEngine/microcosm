"""Invented PUF55 whole-Population attachment and required replay controls.

These tests use the maintained graph, QRF, finalizer and materialized verifier.
They neither admit native sources nor establish production or release acceptance.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_full_puf_enrichment import InventedSource
from test_us_full_puf_output_profiles import InventedProfileMatrix
from test_us_graph_full_puf_enrichment import (
    InventedCompleteBoundary,
    _copy_population,
    _load_artifact,
    _verify,
)
from test_us_puf55_survey_ss_profile import SS, SS_TOTAL, _donor55, _frame, _known55

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
from microcosm.graph import population as population_ops
from microcosm.graph.codecs import load_frame_store

PROFILE = full.PUF55_SURVEY_SS


def _independent_finalized_frame(binding, artifacts):
    """Use the real finalizer without the attachment's column/result builder."""
    finalized, _ = full.finalize_full_puf(
        binding.expected_population.frame,
        binding.donor,
        predictor_known=binding.predictor_known,
        matrix=artifacts["matrix"].payload,
        matrix_producer_key=artifacts["matrix"].producer_key,
        raw_draws={
            target: artifacts[f"raw_{index:03d}"].payload
            for index, target in enumerate(PROFILE.targets)
        },
        apply_state=artifacts["apply_state"].payload,
        training_state=artifacts["training_state"].payload,
        last_model=artifacts["last_model"].payload,
        seed=binding.fit_nodes[0].params["seed"],
        profile=PROFILE,
    )
    return finalized


@pytest.fixture(scope="module")
def attached_puf55(tmp_path_factory):
    root = tmp_path_factory.mktemp("puf55_population_attachment")
    frame, donor = _frame(), _donor55()
    known = _known55(frame)
    prepared = full.prepare_full_puf_inputs(
        frame, donor, predictor_known=known, profile=PROFILE
    )
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
        {"tax_unit": Weights(np.ones(len(table)), WeightKind.DESIGN)},
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
            full.codec.sha(("ordinary-puf55-attachment:" + name).encode()),
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
        InventedProfileMatrix.ref,
        population="matrix_input",
        inputs=(Slice("tax_unit", PROFILE.predictors),),
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
        InventedProfileMatrix(),
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
        profile=PROFILE,
    )
    binding = placement.retain_full_puf_attachment(
        population=upstream,
        expected_population=_copy_population(upstream),
        population_node=boundary,
        input_owners=dict(upstream.owners),
        donor=donor,
        predictor_known=known,
        matrix=matrix,
        fit_nodes=fits,
        apply_nodes=applies,
        profile=PROFILE,
    )
    nodes = placement.full_puf_attachment_nodes(binding)
    kernels.register(placement.FullPufMaskKernel(binding))
    kernels.register(placement.FullPufAttachKernel(binding))
    compiled = compile_graph(
        replace(base_graph, nodes=(*base_graph.nodes, *fits, *applies, *nodes))
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
        actual_upstream, actual = observed[boundary.id], observed[nodes[-1].id]
        evidence = placement.verify_materialized_full_puf_attachment(
            binding,
            upstream_population=actual_upstream,
            population=actual,
            artifacts=artifacts,
            producer_keys=producer_keys,
        )
        # Exercise full Frame storage as well as required column-artifact replay.
        frame_key = full.codec.sha(
            ("puf55-materialized:" + manifest.node(nodes[-1].id).key).encode()
        )
        store.put_frame(frame_key, actual.frame)
        replay.same_replayed_frame(actual.frame, store.load_frame(frame_key))
        results.append(
            SimpleNamespace(
                manifest=manifest,
                upstream=_copy_population(actual_upstream),
                population=_copy_population(actual),
                artifacts=artifacts,
                producer_keys=producer_keys,
                evidence=evidence,
                finalized=_independent_finalized_frame(binding, artifacts),
            )
        )
    replay.same_replayed_population(results[0].population, results[1].population)
    assert results[0].artifacts == results[1].artifacts
    assert results[0].manifest.key == results[1].manifest.key
    return SimpleNamespace(binding=binding, nodes=nodes, fits=fits, results=results)


def test_puf55_cold_and_required_replay_preserve_survey_incumbents(attached_puf55):
    case = attached_puf55
    attach = case.nodes[-1]
    assert len(case.fits) == len(PROFILE.targets) == 55
    assert len(PROFILE.predictors) == 9
    assert PROFILE.predictors == (*full.PUF59.predictors, SS_TOTAL)
    assert tuple(owned.column for owned in attach.outputs) == PROFILE.targets
    assert not set(SS) & {owned.column for owned in attach.outputs}
    assert not set(SS) & set(case.binding.donor)
    owned_cells = {(owned.entity, owned.column) for owned in attach.outputs}
    added_cells = owned_cells | set(placement.MASKS.items())

    for result in case.results:
        evidence = result.evidence
        assert evidence["complete_population_compared"] is True
        assert evidence["output_profile"] == PROFILE.value
        assert evidence["target_order"] == list(PROFILE.targets)
        assert evidence["predictor_order"] == list(PROFILE.predictors)
        assert evidence["target_count"] == 55 and evidence["tail_bounds"]
        assert evidence["source_admission_issued"] is False
        assert evidence["release_eligible"] is False
        decoded = model_input.decode_recipient_matrix(
            result.artifacts["matrix"].payload
        )
        assert tuple(decoded.features) == PROFILE.predictors
        np.testing.assert_array_equal(
            decoded.entity_ids, case.binding.predictor_known.index
        )

        frame, upstream = result.population.frame, result.upstream.frame
        assert frame.schema == upstream.schema and frame.entities == upstream.entities
        assert frame.links == upstream.links and frame.metadata == upstream.metadata
        assert frame.mass_log == upstream.mass_log
        assert result.population.mass_ledger == result.upstream.mass_ledger
        assert result.population.weight_kind == result.upstream.weight_kind
        assert (
            result.population.design_weights.keys()
            == result.upstream.design_weights.keys()
        )
        pd.testing.assert_series_equal(frame.strata, upstream.strata, check_exact=True)
        for entity in upstream.weighted_entities:
            before, after = upstream.weights_for(entity), frame.weights_for(entity)
            assert before.kind is after.kind
            np.testing.assert_array_equal(before.values, after.values)
        for entity, values in result.upstream.design_weights.items():
            np.testing.assert_array_equal(
                values, result.population.design_weights[entity]
            )
        for cell, owner in result.upstream.owners.items():
            if cell not in added_cells:
                assert result.population.owners[cell] == owner
        for entity in upstream.entities:
            before, after = upstream.table(entity), frame.table(entity)
            pd.testing.assert_index_equal(before.index, after.index, exact=True)
            for column in before:
                if (entity, column) not in owned_cells:
                    assert population_ops.storage_equal(before[column], after[column])

        has_nonzero_oracle_value = False
        for owned in attach.outputs:
            before, after = upstream.table(owned.entity), frame.table(owned.entity)
            mask = support.puf_tax_detail_clone_mask(before, entity=owned.entity)
            assert owned.rows == placement.MASKS[owned.entity]
            assert after.loc[mask, owned.column].notna().all()
            assert result.population.owners[(owned.entity, owned.column)] == attach.id
            # Align by entity IDs, independently of the attachment's positional
            # selection and column construction. Replay alone shares that code.
            id_column = upstream.schema.entity_id_column(owned.entity)
            entity_ids = pd.Index(before.loc[mask, id_column], name=id_column)
            expected = (
                result.finalized.table(owned.entity)
                .set_index(id_column, verify_integrity=True)
                .loc[entity_ids, owned.column]
                .astype(owned.dtype)
            )
            actual = after.set_index(id_column, verify_integrity=True).loc[
                entity_ids, owned.column
            ]
            pd.testing.assert_series_equal(actual, expected, check_exact=True)
            has_nonzero_oracle_value |= bool(expected.ne(0).any())
            if owned.column in before:
                assert population_ops.storage_equal(
                    before[owned.column], after[owned.column], ~mask
                )
            else:
                assert after.loc[~mask, owned.column].isna().all()
        # Keep a nontrivial oracle: replacing all selected outputs with zeros
        # must be detected even when the kernel and replay verifier agree.
        assert has_nonzero_oracle_value

        people = upstream.table("person")
        mask = support.puf_tax_detail_clone_mask(people, entity="person")
        for component in SS:
            # Nonzero incumbents and unknownness exist on both support arms.
            for selected in (mask, ~mask):
                assert people.loc[selected, component].isna().any()
                assert people.loc[selected, component].gt(0).any()
            np.testing.assert_array_equal(
                people[component].to_numpy().view("uint64"),
                frame.table("person")[component].to_numpy().view("uint64"),
            )
        np.testing.assert_array_equal(
            upstream.table("tax_unit")[SS_TOTAL].to_numpy().view("uint64"),
            frame.table("tax_unit")[SS_TOTAL].to_numpy().view("uint64"),
        )


@pytest.mark.parametrize("mutation", ("observed_value", "unknown_to_zero"))
def test_puf55_required_population_verifier_rejects_ss_changes(
    attached_puf55, mutation
):
    case, result = attached_puf55, attached_puf55.results[-1]
    changed = _copy_population(result.population)
    people = changed.frame.table("person")
    mask = support.puf_tax_detail_clone_mask(people, entity="person")
    component = SS[0]
    selected = mask & (
        people[component].gt(0)
        if mutation == "observed_value"
        else people[component].isna()
    )
    row = people.index[selected][0]
    people.loc[row, component] = (
        people.loc[row, component] + 1.0 if mutation == "observed_value" else 0.0
    )
    # SS incumbents are outside the model surface. Equal predictors cannot excuse
    # changing their values or converting an unknown recipient benefit to zero.
    matrices = tuple(
        full.prepare_full_puf_inputs(
            population.frame,
            case.binding.donor,
            predictor_known=case.binding.predictor_known,
            profile=PROFILE,
        ).matrix
        for population in (result.population, changed)
    )
    assert matrices[0] == matrices[1]
    with pytest.raises(ValueError, match="SURVEY_POPULATION_REPLAY_"):
        _verify(case, result, population=changed)
