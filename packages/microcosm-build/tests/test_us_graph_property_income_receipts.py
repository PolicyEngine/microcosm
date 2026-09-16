"""Receipt reconstruction over the real, small invented property graph."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_property_income import FEATURES, assert_preserved, execute, setup

from microcosm.build.us_runtime.graph_property_income_receipts import (
    verify_property_model_receipts,
)
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import graph_legacy_train, qrf, qrf_target
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph.population import Population


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPULACE_FIT_N_JOBS", "1")
        patch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
        values = setup(tmp_path_factory.mktemp("property-receipts"))
        compiled, store, _, _, donor, recipient, _ = values
        cold = execute(values)
        replay = execute(values, resume="require")
        assert len(compiled.order) == 12
        assert all(node.hit for node in replay.nodes.values())
        nodes = compiled.graph.nodes
        artifacts = {
            (node_id, name): store.load_bytes(key)
            for node_id, node in cold.nodes.items()
            for name, key in node.opaque_artifacts.items()
        }
        replay_artifacts = {
            (node_id, name): store.load_bytes(key)
            for node_id, node in replay.nodes.items()
            for name, key in node.opaque_artifacts.items()
        }
        yield SimpleNamespace(
            nodes=nodes,
            donor=Population.from_frame(donor, "donor"),
            recipient=Population.from_frame(recipient, "recipient"),
            artifacts=artifacts,
            replay_artifacts=replay_artifacts,
            cold=cold,
            replay=replay,
        )


def verify(case, **changes):
    args = {
        "nodes": case.nodes,
        "donor_population": case.donor,
        "recipient_population": case.recipient,
        "artifacts": case.artifacts,
    }
    return verify_property_model_receipts(**{**args, **changes})


def changed_population(population, *, person=None, weights=None):
    frame = population.frame
    changed = Frame(
        {
            entity: person
            if entity == "person" and person is not None
            else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {
            entity: weights if weights is not None else frame.weights_for(entity)
            for entity in frame.weighted_entities
        },
        metadata=frame.metadata,
    )
    return Population.from_frame(changed, population.version)


def changed_packet(case, key, change):
    artifacts = dict(case.artifacts)
    packet = codec.decode_json(artifacts[key])
    change(packet)
    artifacts[key] = codec.encode_json(packet)
    return artifacts


def test_actual_cold_required_and_published_receipts_match_without_fitting(
    case, monkeypatch
):
    def no_fit(*args, **kwargs):
        raise AssertionError("Receipt verification must never fit a model")

    monkeypatch.setattr(qrf_target, "fit_target", no_fit)
    monkeypatch.setattr(graph_legacy_train, "fit_target", no_fit)
    monkeypatch.setattr(qrf.RegimeGatedQRF, "fit_draw_next", no_fit)
    calls = []
    original_apply = qrf_target.apply_target

    def apply(*args, **kwargs):
        calls.append(args[0].target)
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(qrf_target, "apply_target", apply)
    result = verify(case)
    assert len(calls) == 4
    assert len(result) == 8
    for manifest in (case.cold, case.replay):
        # The executor adds capabilities/input-writer provenance around the raw
        # kernel receipt. The owning host's _states validates that outer layer.
        assert result == {
            node_id: {
                key: value
                for key, value in manifest.node(node_id).receipt.items()
                if key != "capabilities"
            }
            for node_id in result
        }
    assert all(case.cold.node(n).receipt == case.replay.node(n).receipt for n in result)
    assert verify(case, artifacts=case.replay_artifacts) == result
    assert len(calls) == 8
    assert_preserved(case.donor.frame, case.cold.population("donor"))
    assert_preserved(case.recipient.frame, case.cold.population("recipient"))
    assert case.artifacts == case.replay_artifacts
    # Table row labels and exact int64 IDs above 2**53 are separate axes.
    assert case.recipient.frame.person.index.tolist() == [9, 2, 17, 41]
    assert (case.recipient.frame.person.person_id > 2**53).all()
    assert case.donor.frame.weights_for("household").values[-1] == 0


@pytest.mark.parametrize("branch", ["donor", "recipient"])
@pytest.mark.parametrize("mutation", ["feature", "row_order", "float32", "nonfinite"])
def test_changed_population_inputs_refuse(case, branch, mutation):
    population = getattr(case, branch)
    person = population.frame.person.copy()
    if mutation == "feature":
        person.loc[:, FEATURES[0]] += 1e12
        person.loc[:, "age"] += 1e6
        if branch == "donor":
            # Keep the declared property sum valid while changing consumed bytes.
            person.loc[:, "property_broad_receipts"] += 1e12
    elif mutation == "row_order":
        person.index = pd.Index(list(reversed(person.index)), name=person.index.name)
    elif mutation == "float32":
        person[FEATURES[0]] = person[FEATURES[0]].astype("float32")
    else:
        person.loc[person.index[0], FEATURES[0]] = np.nan
    changed = changed_population(population, person=person)
    with pytest.raises(ValueError):
        verify(case, **{branch + "_population": changed})


@pytest.mark.parametrize("branch", ["donor", "recipient"])
@pytest.mark.parametrize("mutation", ["id_dtype", "duplicate_id"])
def test_person_identity_invalidity_refuses_separately(case, branch, mutation):
    # Exercise a post-construction table corruption; a normal Frame constructor
    # also refuses invalid identities before this helper is reached.
    population = changed_population(getattr(case, branch))
    person = population.frame.person
    if mutation == "id_dtype":
        person["person_id"] = person.person_id.astype("float64")
    else:
        person.loc[person.index[1], "person_id"] = person.person_id.iloc[0]
    with pytest.raises(ValueError, match="PERSON_IDENTITY_OR_VALUES"):
        verify(case, **{branch + "_population": population})


@pytest.mark.parametrize("mutation", ["values", "kind"])
def test_changed_original_design_weights_refuse(case, mutation):
    original = case.donor.frame.weights_for("household")
    weights = Weights(
        original.values + (1 if mutation == "values" else 0),
        WeightKind.CALIBRATED if mutation == "kind" else WeightKind.DESIGN,
    )
    donor = changed_population(case.donor, weights=weights)
    with pytest.raises(ValueError):
        verify(case, donor_population=donor)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_fit",
        "missing_apply",
        "duplicate",
        "swapped_fits",
        "swapped_applies",
        "wrong_seed",
        "wrong_phase",
        "wrong_features",
        "wrong_target",
        "wrong_edge",
        "wrong_population",
        "unexpected_param",
    ],
)
def test_malformed_declarations_refuse(case, mutation):
    nodes = list(case.nodes)
    fit_index = next(i for i, n in enumerate(nodes) if n.id == "property.fit.000")
    apply_index = next(i for i, n in enumerate(nodes) if n.id == "property.apply.000")
    if mutation == "missing_fit":
        nodes.pop(fit_index)
    elif mutation == "missing_apply":
        nodes.pop(apply_index)
    elif mutation == "duplicate":
        nodes.append(nodes[fit_index])
    elif mutation == "swapped_fits":
        nodes[fit_index], nodes[fit_index + 1] = nodes[fit_index + 1], nodes[fit_index]
    elif mutation == "swapped_applies":
        nodes[apply_index], nodes[apply_index + 1] = (
            nodes[apply_index + 1],
            nodes[apply_index],
        )
    elif mutation == "wrong_edge":
        node = nodes[apply_index]
        nodes[apply_index] = replace(
            node,
            artifact_inputs=(
                replace(node.artifact_inputs[0], producer="property.fit.001"),
                *node.artifact_inputs[1:],
            ),
        )
    elif mutation == "wrong_population":
        nodes[apply_index] = replace(nodes[apply_index], population="donor")
    else:
        index = apply_index if mutation == "wrong_seed" else fit_index + 1
        node = nodes[index]
        field, value = {
            "wrong_seed": ("seed", True),
            "wrong_phase": ("phase", "foreign"),
            "wrong_features": ("predictors", tuple(reversed(FEATURES))),
            "wrong_target": ("target", "property_dividends"),
            "unexpected_param": ("extra", 1),
        }[mutation]
        nodes[index] = replace(node, params={**node.params, field: value})
    with pytest.raises(ValueError):
        verify(case, nodes=tuple(nodes))


@pytest.mark.parametrize(
    "node,output",
    [
        ("property.fit.000", "model"),
        ("property.fit.003", "training_state"),
        ("property.apply.000", "raw_draw"),
        ("property.apply.003", "apply_state"),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "not_bytes"])
def test_missing_or_unowned_payload_shape_refuses(case, node, output, mutation):
    artifacts = dict(case.artifacts)
    if mutation == "missing":
        del artifacts[node, output]
    else:
        artifacts[node, output] = bytearray(artifacts[node, output])
    with pytest.raises(ValueError, match="ARTIFACT_ROSTER"):
        verify(case, artifacts=artifacts)


@pytest.mark.parametrize(
    "mutation",
    [
        "model_bytes",
        "model_swapped",
        "training_id",
        "training_donor_state",
        "training_history",
        "apply_model_history",
        "apply_seed",
        "apply_draw_rng",
        "raw_and_hash",
    ],
)
def test_altered_artifacts_refuse(case, mutation):
    if mutation in ("model_bytes", "model_swapped"):
        artifacts = dict(case.artifacts)
        key = ("property.fit.000", "model")
        artifacts[key] = (
            artifacts[key][:-1] + bytes([artifacts[key][-1] ^ 1])
            if mutation == "model_bytes"
            else artifacts["property.fit.001", "model"]
        )
    elif mutation == "training_id":
        artifacts = changed_packet(
            case,
            ("property.fit.001", "training_state"),
            lambda p: p["models"][-1].update(training_id="0" * 64),
        )
    elif mutation == "training_donor_state":
        artifacts = changed_packet(
            case,
            ("property.fit.000", "training_state"),
            lambda p: p["state"].update(weight_sha256="0" * 64),
        )
    elif mutation == "training_history":
        artifacts = changed_packet(
            case,
            ("property.fit.001", "training_state"),
            lambda p: p["models"][0].update(sha256="0" * 64),
        )
    elif mutation == "apply_model_history":
        artifacts = changed_packet(
            case,
            ("property.apply.001", "apply_state"),
            lambda p: p["models"][0].update(sha256="0" * 64),
        )
    elif mutation == "apply_seed":
        artifacts = changed_packet(
            case, ("property.apply.000", "apply_state"), lambda p: p.update(seed=999)
        )
    elif mutation == "apply_draw_rng":
        artifacts = changed_packet(
            case,
            ("property.apply.000", "apply_state"),
            lambda p: p["state"].update(draw_rng_state=np.random.PCG64(912).state),
        )
    else:
        key = ("property.apply.000", "raw_draw")
        raw = codec.read_raw_target(
            case.artifacts[key],
            target="property_ordinary_interest",
            index=case.recipient.frame.person.index,
        )
        payload = codec.encode_raw_target(
            raw + 1,
            target="property_ordinary_interest",
            index=case.recipient.frame.person.index,
        )
        artifacts = changed_packet(
            case,
            ("property.apply.000", "apply_state"),
            lambda p: p["raw_targets"][0].update(sha256=codec.sha(payload)),
        )
        artifacts[key] = payload
    with pytest.raises(ValueError):
        verify(case, artifacts=artifacts)
