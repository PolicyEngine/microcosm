"""Ordinary invented source control; real QRF, graph/store, and materialized checks.

This does not qualify genuine PUF donors, run a tax engine, or certify a release.
The donor CREATE below is deliberately test-only; no production source issuer or
readiness function is mocked. The survey helper uses its existing invented pins.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_survey_population import authenticated_arguments

from microcosm.build.us_runtime import graph_current_survey_puf_transfer as transfer
from microcosm.build.us_runtime import graph_puf_diagnostic_consumer as host
from microcosm.build.us_runtime import graph_survey_population as survey
from microcosm.build.us_runtime import survey_population_replay as replay
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
    source_hash,
)
from microcosm.graph.codecs import load_frame_store
from microcosm.graph.keys import opaque_artifact_key

codec, detail = transfer.codec, transfer.detail


def _invented_donor():
    ids = np.arange(101, 113, dtype=np.int64)
    mars = np.tile(np.arange(1, 5), 3)
    return Frame(
        {
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": ids,
                    "reported_wage_proxy": np.array(
                        [
                            0,
                            30000,
                            48000,
                            60000,
                            5000,
                            20000,
                            40000,
                            90000,
                            12000,
                            35000,
                            55000,
                            75000,
                        ],
                        dtype="float64",
                    ),
                    **{f"mars_{i}": (mars == i).astype("float64") for i in range(1, 5)},
                    "E00900": np.array(
                        [
                            -2400,
                            0,
                            1200,
                            3600,
                            -600,
                            0,
                            800,
                            2400,
                            -1200,
                            0,
                            600,
                            4800,
                        ],
                        dtype="float64",
                    ),
                }
            ),
            "person": pd.DataFrame({"person_id": ids, "person_tax_unit_id": ids}),
        },
        EntitySchema(group_entities=("tax_unit",)),
        {"tax_unit": Weights(np.arange(1, 13, dtype="float64"), WeightKind.DESIGN)},
        metadata={
            "scope": "invented_donor_mechanism_only",
            "genuine_source_admitted": False,
        },
    )


class InventedDonorKernel(KernelBase):
    ref = "test.current_survey_puf.invented_donor@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            _invented_donor, type(self), dependencies=self.capabilities.dependencies
        )

    def run(self, context):
        assert not context.node.inputs and not context.artifacts
        assert context.node.sources == ("test.current_donor_source",)
        return KernelResult(
            frame=load_frame_store(context.sources["test.current_donor_source"])
        )


def _copy_population(population, *, metadata=None):
    frame = population.frame
    copied = Frame(
        {e: frame.table(e).copy(deep=True) for e in frame.entities},
        frame.schema,
        {
            e: Weights(frame.weights_for(e).values.copy(), frame.weights_for(e).kind)
            for e in frame.weighted_entities
        },
        frame.strata.copy(deep=True),
        metadata=frame.metadata if metadata is None else metadata,
        mass_log=frame.mass_log,
    )
    return replace(
        population,
        frame=copied,
        owners=dict(population.owners),
        weight_kind=dict(population.weight_kind),
        design_weights={e: v.copy() for e, v in population.design_weights.items()},
    )


def test_current_survey_real_fit_draw_attach_and_complete_replay(tmp_path, monkeypatch):
    live = survey.run_authenticated_survey_population(
        **authenticated_arguments(tmp_path, monkeypatch),
        clones=True,
        return_values=True,
    )
    preparation, allocated, clone = (
        live.preparation,
        live.allocated_population,
        live.clone_population,
    )
    projection, matrix, mask, _ = host.qualify_current_survey_host(
        preparation, allocated, clone
    )
    document = codec.decode_json(projection)
    pins = {}
    for edge in host.current_survey_host_edges():
        parent = live.manifest.node(edge.producer)
        key = parent.opaque_artifacts[edge.artifact]
        pins[edge.name] = {
            "producer_key": parent.key,
            "artifact_key": key,
            "payload_sha256": codec.sha(live.store.load_bytes(key)),
        }
    host_nodes = host.current_survey_host_nodes(
        frame_column_declarations(clone.frame),
        host_pins=pins,
        **{
            k: document[k]
            for k in ("preparation_sha256", "allocation_sha256", "clone_sha256")
        },
    )
    donor_frame = _invented_donor()
    donor_key = codec.sha(
        donor_frame.table("tax_unit").to_json(orient="table").encode()
        + donor_frame.weights_for("tax_unit").values.tobytes()
    )
    donor_path = live.store.put_frame(donor_key, donor_frame)
    live.store.codecs.register("frame-store", load_frame_store)
    sources = {**live.sources, "test.current_donor_source": donor_path}
    donor = Node(
        "test.current_donor",
        InventedDonorKernel.ref,
        sources=("test.current_donor_source",),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned("tax_unit", name, "float64")
            for name in (*detail.FEATURES, detail.TARGET)
        ),
    )
    fits = transfer.legacy_qrf_train_nodes(
        "test.current_fit",
        population=donor.id,
        entity="tax_unit",
        predictors=detail.FEATURES,
        targets=(detail.TARGET,),
        seed=transfer.SEED,
        phase=transfer.SCOPE,
        n_estimators=2,
        zero_atol=0,
    )
    nodes = transfer.current_survey_transfer_nodes(clone.frame, fit_nodes=fits)
    # A reduced/changed donor specification is not silently used as this recipe.
    with pytest.raises(ValueError, match="SURVEY_TRANSFER_FIT_DECLARATION"):
        transfer.current_survey_transfer_nodes(
            clone.frame,
            fit_nodes=(replace(fits[0], params={**dict(fits[0].params), "seed": 579}),),
        )
    compiled = compile_graph(
        replace(
            live.compiled.graph,
            sources=(
                *live.compiled.graph.sources,
                SourceRef("test.current_donor_source", "frame-store"),
            ),
            nodes=(*live.compiled.graph.nodes, *host_nodes, donor, *fits, *nodes),
        )
    )
    for kernel in (
        host.CurrentSurveyHostProjectionKernel(preparation, allocated, clone),
        host.CurrentSurveyMatrixKernel(),
        InventedDonorKernel(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
        transfer.CurrentSurveyPlacementKernel(preparation, allocated, clone),
        transfer.CurrentSurveyAttachKernel(preparation, allocated, clone),
    ):
        live.kernels.register(kernel)
    artifacts = (
        (
            host.CURRENT_PROJECTION_NODE,
            "projection",
            host.CURRENT_PROJECTION_TYPE,
            host.CurrentSurveyHostProjectionKernel,
        ),
        (
            host.CURRENT_MATRIX_NODE,
            "matrix",
            transfer.model_input.RECIPIENT_MATRIX_TYPE,
            host.CurrentSurveyMatrixKernel,
        ),
        (
            transfer.PLACEMENT_NODE,
            "placement",
            transfer.PLACEMENT_TYPE,
            transfer.CurrentSurveyPlacementKernel,
        ),
        (
            transfer.APPLY_PREFIX + ".000",
            "raw_draw",
            codec.RAW_TARGET_TYPE,
            LegacyQRFApplyMatrixKernel,
        ),
        (
            transfer.APPLY_PREFIX + ".000",
            "apply_state",
            transfer.MATRIX_APPLY_STATE_TYPE,
            LegacyQRFApplyMatrixKernel,
        ),
    )
    results = []
    for resume in ("auto", "require"):
        observed = {}
        manifest = run_graph(
            compiled,
            store=live.store,
            kernels=live.kernels,
            sources=sources,
            resume=resume,
            _population_observer=lambda name, population, observed=observed: (
                observed.__setitem__(name, population)
            ),
        )
        loaded = {}
        for node_id, name, type_, kernel in artifacts:
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
                capabilities=kernel.capabilities,
            )
            loaded[name] = payload
        assert loaded["projection"] == projection and loaded["matrix"] == matrix
        producer_key = manifest.node(host.CURRENT_MATRIX_NODE).key
        actual = observed[transfer.ATTACH_NODE]
        verification = dict(
            population=actual,
            **loaded,
            matrix_producer_key=producer_key,
        )
        # Ordinary nodes store their columns, not a complete frame_key. Persist
        # the observed final Frame explicitly through the maintained store API.
        saved_key = codec.sha(
            ("test-final-frame:" + manifest.node(transfer.ATTACH_NODE).key).encode()
        )
        live.store.put_frame(saved_key, actual.frame)
        replay.same_replayed_frame(actual.frame, live.store.load_frame(saved_key))
        checked = transfer.verify_materialized_current_survey_transfer(
            preparation,
            allocated,
            clone,
            **verification,
        )
        assert checked["release_eligible"] is False
        assert checked["source_admission"] == detail.SOURCE_ADMISSION
        table = actual.frame.table("tax_unit")
        np.testing.assert_array_equal(table[detail.MASK], mask)
        assert table.loc[~mask, detail.OUTPUT].isna().all()
        expected = transfer._raw_values(
            matrix, producer_key, loaded["raw_draw"], loaded["apply_state"]
        )
        np.testing.assert_array_equal(
            table.loc[mask, detail.OUTPUT].to_numpy(), expected.to_numpy()
        )
        assert np.isfinite(table.loc[mask, detail.OUTPUT]).all()
        assert np.any(clone.frame.resolve_weights("tax_unit").values[mask] == 0)
        assert len(expected) == int(mask.sum())  # Zero-weight recipients retained.
        results.append((loaded, _copy_population(actual)))

        # Value-valid changed raw bytes must fail application binding, not an issuer stub.
        changed = expected.to_numpy().copy()
        changed[0] += 1
        prepared = transfer.model_input.decode_recipient_matrix(matrix)
        changed_raw = codec.encode_raw_target(
            changed, target=detail.TARGET, index=prepared.features.index
        )
        with pytest.raises(ValueError, match="SURVEY_TRANSFER_RAW_STATE"):
            transfer.verify_materialized_current_survey_transfer(
                preparation,
                allocated,
                clone,
                **{**verification, "raw_draw": changed_raw},
            )
        with pytest.raises(ValueError, match="SURVEY_TRANSFER_PLACEMENT_BINDING"):
            transfer.verify_materialized_current_survey_transfer(
                preparation,
                allocated,
                clone,
                **{**verification, "matrix_producer_key": "f" * 64},
            )
        for mutation in (
            "metadata",
            "native_output",
            "clone_output",
            "mask",
            "owner",
            "strata",
        ):
            changed_population = _copy_population(
                actual,
                metadata={**actual.frame.metadata, "unexpected": "changed"}
                if mutation == "metadata"
                else None,
            )
            if mutation == "native_output":
                changed_population.frame.table("tax_unit").loc[~mask, detail.OUTPUT] = (
                    0.0
                )
            elif mutation == "clone_output":
                changed_table = changed_population.frame.table("tax_unit")
                changed_table.loc[changed_table.index[mask][0], detail.OUTPUT] += 1.0
            elif mutation == "mask":
                changed_population.frame.table("tax_unit").loc[:, detail.MASK] = False
            elif mutation == "owner":
                object.__setattr__(
                    changed_population,
                    "owners",
                    {
                        **changed_population.owners,
                        ("tax_unit", detail.OUTPUT): "wrong.owner",
                    },
                )
            elif mutation == "strata":
                changed_population.frame.strata.name = "wrong_name"
            reason = (
                "^SURVEY_POPULATION_REPLAY_NATIVE_BITS$"
                if mutation == "clone_output"
                else None
            )
            with pytest.raises(ValueError, match=reason):
                transfer.verify_materialized_current_survey_transfer(
                    preparation,
                    allocated,
                    clone,
                    **{**verification, "population": changed_population},
                )
    assert results[0][0] == results[1][0]
    replay.same_replayed_population(results[0][1], results[1][1])
