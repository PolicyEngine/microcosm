"""Split model kernels fit once and apply by stable recipient identity."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import fit
from microcosm.fit.graph_models import (
    QRF_MODEL_TYPE,
    QRFApplyKernel,
    QRFTrainKernel,
    load_qrf_model,
)
from microcosm.frame import WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactValue,
    KernelContext,
    Node,
    NumericScope,
    Owned,
    Slice,
    keyed_uniform,
)

STREAM = ("sha256-u53-v1", "test-qrf", 0, 19)


def context(node, table, artifact=None, weights=None):
    return KernelContext(
        node=node,
        tables={"household": table},
        weights={
            "household": Weights(
                np.ones(len(table)) if weights is None else weights,
                WeightKind.DESIGN,
            )
        },
        strata=pd.Series(dtype="string"),
        params=node.params,
        rng=np.random.default_rng(9),
        artifacts={} if artifact is None else {"model": artifact},
    )


@pytest.fixture(scope="module")
def trained():
    donor = pd.DataFrame(
        {
            "household_id": np.arange(60),
            "x": np.arange(60) % 3,
            "y": np.arange(60) + 1.0,
        }
    )
    node = Node(
        "train",
        QRFTrainKernel.ref,
        inputs=(Slice("household", ("x", "y")),),
        artifact_outputs=(ArtifactOutput("model", QRF_MODEL_TYPE),),
        params={"predictors": ("x",), "targets": ("y",), "n_estimators": 4, "seed": 8},
    )
    result = QRFTrainKernel().run(context(node, donor))
    artifact = ArtifactValue(
        result.artifacts["model"], QRF_MODEL_TYPE, "a" * 64, "b" * 64, NumericScope()
    )
    return node, donor, result, artifact


def apply_context(table, artifact):
    node = Node(
        "apply",
        QRFApplyKernel.ref,
        inputs=(Slice("household", ("x",)),),
        outputs=(Owned("household", "y", "float64"),),
        artifact_inputs=(ArtifactInput("model", "train", "model", QRF_MODEL_TYPE),),
        params={"random_stream": STREAM, "period": 2025},
    )
    return context(node, table, artifact)


def test_model_matches_public_fit_and_stateless_apply(trained):
    node, donor, result, artifact = trained
    model = load_qrf_model(artifact)
    direct = fit(
        donor[["x", "y"]],
        ["x"],
        ["y"],
        weights=np.ones(len(donor)),
        n_estimators=4,
        seed=8,
    )
    assert model.regimes() == direct.regimes()
    assert result.receipt["source_weight_kind"] == "design"
    assert result.receipt["fit_weight_kind"] == "explicit"
    table = pd.DataFrame({"household_id": [21, 32, 87], "x": [0, 1, 2]})
    actual = (
        QRFApplyKernel().run(apply_context(table, artifact)).columns[("household", "y")]
    )
    arrays = {
        kind: {
            "y": keyed_uniform(
                stream=STREAM,
                keys=[(i, "qrf", "y", 2025, kind) for i in table.household_id],
            )
        }
        for kind in ("quantiles", "sign_uniforms")
    }
    expected = direct.predict_from_uniforms(table, **arrays)
    np.testing.assert_array_equal(actual.values, expected.y.values)


def test_reordering_and_unrelated_recipient_leave_draws_unchanged(trained):
    artifact = trained[-1]
    table = pd.DataFrame({"household_id": [21, 32, 87], "x": [0, 1, 2]})
    kernel = QRFApplyKernel()
    first = kernel.run(apply_context(table, artifact)).columns[("household", "y")]
    added = pd.concat([pd.DataFrame({"household_id": [1], "x": [1]}), table.iloc[::-1]])
    second = kernel.run(apply_context(added, artifact)).columns[("household", "y")]
    pd.testing.assert_series_equal(first, second.loc[first.index])


def test_zero_weight_support_excluded_before_regime_fit(trained):
    node, donor, _, _ = trained
    donor = donor.copy()
    donor.loc[0, "y"] = -999.0
    weights = np.ones(len(donor))
    weights[0] = 0
    result = QRFTrainKernel().run(context(node, donor, weights=weights))
    assert result.receipt["regimes"] == {"y": "positive_only"}
    assert result.receipt["excluded_zero_weight_rows"] == 1


@pytest.mark.parametrize("mutation", ["type", "payload"])
def test_invalid_model_artifact_rejected(trained, mutation):
    artifact = trained[-1]
    if mutation == "payload":
        artifact = replace(artifact, payload=b"bad model")
    else:
        from microcosm.graph import ArtifactType

        artifact = replace(artifact, type=ArtifactType("other", 1))
    with pytest.raises(ValueError, match="model|artifact"):
        load_qrf_model(artifact)


def test_training_honors_unequal_effective_weights(trained):
    node, donor, _, _ = trained
    donor = donor.copy()
    donor["x"] = 1
    donor["y"] = np.where(np.arange(len(donor)) % 2, 100.0, 1.0)
    weights = np.where(donor.y == 1, 1000.0, 1.0)
    result = QRFTrainKernel().run(context(node, donor, weights=weights))
    artifact = ArtifactValue(
        result.artifacts["model"], QRF_MODEL_TYPE, "a" * 64, "b" * 64, NumericScope()
    )
    recipient = pd.DataFrame({"household_id": np.arange(200), "x": 1})
    weighted = QRFApplyKernel().run(apply_context(recipient, artifact))
    assert weighted.columns[("household", "y")].mean() < 3.0
    assert result.receipt["donor_weight_sum"] == weights.sum()


def test_distinct_streams_make_distinct_applications(trained):
    recipient = pd.DataFrame({"household_id": np.arange(100), "x": 1})
    first = apply_context(recipient, trained[-1])
    node = replace(
        first.node,
        params={"random_stream": (*STREAM[:2], 1, STREAM[3]), "period": 2025},
    )
    second = replace(first, node=node, params=node.params)
    first_draws = QRFApplyKernel().run(first).columns[("household", "y")]
    second_draws = QRFApplyKernel().run(second).columns[("household", "y")]
    assert not np.array_equal(first_draws, second_draws)


@pytest.mark.parametrize(
    "module",
    [
        QRFApplyKernel.__module__,
        "microcosm.graph.randomness",
        "microcosm.graph.canonical",
    ],
)
def test_application_code_change_preserves_training_identity(monkeypatch, module):
    train = QRFTrainKernel()
    apply = QRFApplyKernel()
    fit_before = train.implementation_hash()
    apply_before = apply.implementation_hash()
    application_source = Path(import_module(module).__file__).resolve()
    original = Path.read_bytes

    def changed_source(path):
        payload = original(path)
        return (
            payload + b"\n# application-only change\n"
            if path.resolve() == application_source
            else payload
        )

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    assert train.implementation_hash() == fit_before
    assert apply.implementation_hash() != apply_before
