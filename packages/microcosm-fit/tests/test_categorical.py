"""Tiny real classifier contracts; no survey source or SS beneficiary claim."""

import hashlib
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from microcosm.fit import categorical as cat
from microcosm.frame import WeightKind, Weights

CLASSES = ("survivors", "retirement", "disability", "dependents")
SOURCE = hashlib.sha256(b"invented categorical donor source").hexdigest()


def donor():
    index = pd.Index(np.arange(8, dtype=np.int64) + 2**53 + 1, name="person_id")
    features = pd.DataFrame({"age": np.zeros(8), "total": np.zeros(8)}, index=index)
    labels = pd.Series(CLASSES * 2, index=index)
    weights = Weights(np.array([1.0, 2.0, 3.0, 4.0] * 2), WeightKind.DESIGN)
    return features, labels, weights


def fit(features=None, labels=None, weights=None, **changes):
    x, y, w = donor()
    options = dict(
        weights=w if weights is None else weights,
        classes=CLASSES,
        entity="person",
        source_sha256=SOURCE,
        seed=101,
        config=cat.CategoricalConfig(max_iter=2),
    )
    options.update(changes)
    return cat.fit_categorical(
        x if features is None else features, y if labels is None else labels, **options
    )


@pytest.fixture(scope="module")
def model():
    return fit()


def test_real_design_weighted_fit_preserves_classes_and_exact_declared_order(model):
    x, y, w = donor()
    before = x.copy(deep=True)
    result = model.probabilities(x.iloc[:2], entity="person")
    assert tuple(result.columns) == CLASSES
    np.testing.assert_allclose(
        result.to_numpy(), np.tile([0.1, 0.2, 0.3, 0.4], (2, 1)), rtol=0, atol=1e-12
    )
    raw = pickle.loads(model._pickle)
    assert type(raw) is HistGradientBoostingClassifier
    assert tuple(raw.classes_) != CLASSES
    expected = raw.predict_proba(x.iloc[:2].to_numpy())[
        :, [list(raw.classes_).index(c) for c in CLASSES]
    ]
    np.testing.assert_array_equal(result.to_numpy(), expected)
    pd.testing.assert_frame_equal(x, before)
    assert model.metadata["class_weight_mass"] == [2.0, 4.0, 6.0, 8.0]
    assert model.metadata["weight_kind"] == "design"
    assert model.metadata["rows"] == len(y) == len(w)
    assert model.metadata["implementation"]["sources"]["microcosm.frame.weights"] == (
        hashlib.sha256(Path(cat.frame_weights.__file__).read_bytes()).hexdigest()
    )


def test_trusted_roundtrip_and_recipient_permutation_use_no_draw_rng(model):
    payload = model.to_bytes()
    loaded = cat.CategoricalModel.from_trusted_bytes(
        payload, expected_sha256=cat._sha(payload)
    )
    assert loaded.to_bytes() == payload
    x, _, _ = donor()
    left = loaded.probabilities(x, entity="person")
    right = loaded.probabilities(x.iloc[::-1], entity="person")
    pd.testing.assert_frame_equal(left, right.loc[left.index])
    assert loaded.to_bytes() == payload
    changed = loaded.metadata
    changed["classes"][0] = "changed"
    assert loaded.metadata["classes"] == list(CLASSES)


def test_zero_weight_class_refuses_but_rare_positive_class_survives():
    x, y, w = donor()
    values = w.values.copy()
    values[[0, 4]] = 0
    with pytest.raises(ValueError, match="CLASS_SUPPORT"):
        fit(weights=Weights(values, WeightKind.DESIGN))
    values[0] = 1e-8
    fitted = fit(weights=Weights(values, WeightKind.DESIGN))
    assert fitted.metadata["class_weight_mass"][0] == 1e-8
    assert set(pickle.loads(fitted._pickle).classes_) == set(CLASSES)


@pytest.mark.parametrize(
    "defect",
    [
        "classes_empty",
        "classes_single",
        "classes_duplicate",
        "classes_missing",
        "classes_extra",
        "label_missing",
        "label_index",
        "weight_kind",
        "weight_length",
        "weight_infinite_total",
        "float_ids",
        "duplicate_ids",
        "float32",
        "nonfinite",
        "bool_seed",
        "negative_seed",
        "bad_source",
    ],
)
def test_invalid_training_refuses_before_estimator_fit(monkeypatch, defect):
    x, y, w = donor()
    options = {}
    if defect == "classes_empty":
        options["classes"] = ()
    elif defect == "classes_single":
        options["classes"] = ("one",)
    elif defect == "classes_duplicate":
        options["classes"] = (*CLASSES, CLASSES[0])
    elif defect == "classes_missing":
        options["classes"] = CLASSES[:-1]
    elif defect == "classes_extra":
        options["classes"] = (*CLASSES, "extra")
    elif defect == "label_missing":
        y.iloc[0] = None
    elif defect == "label_index":
        y.index = y.index[::-1]
    elif defect == "weight_kind":
        w = Weights(w.values, WeightKind.IMPORTANCE)
    elif defect == "weight_length":
        w = Weights(w.values[:-1], WeightKind.DESIGN)
    elif defect == "weight_infinite_total":
        w = Weights(np.full(8, 1e308), WeightKind.DESIGN)
    elif defect == "float_ids":
        x.index = x.index.astype(float)
    elif defect == "duplicate_ids":
        x.index = pd.Index(np.zeros(8, dtype=np.int64))
    elif defect == "float32":
        x = x.astype(np.float32)
    elif defect == "nonfinite":
        x.iloc[0, 0] = np.nan
    elif defect == "bool_seed":
        options["seed"] = True
    elif defect == "negative_seed":
        options["seed"] = -1
    else:
        options["source_sha256"] = "bad"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid training reached real fitter")

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", forbidden)
    with pytest.raises(ValueError):
        fit(x, y, w, **options)
    assert not calls


@pytest.mark.parametrize(
    "defect", ["predictor_order", "nonfinite", "entity", "float_ids"]
)
def test_recipient_schema_refuses_before_prediction(model, monkeypatch, defect):
    x, _, _ = donor()
    entity = "person"
    if defect == "predictor_order":
        x = x[x.columns[::-1]]
    elif defect == "nonfinite":
        x.iloc[0, 0] = np.inf
    elif defect == "entity":
        entity = "household"
    else:
        x.index = x.index.astype(float)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid recipients reached prediction")

    monkeypatch.setattr(HistGradientBoostingClassifier, "predict_proba", forbidden)
    with pytest.raises(ValueError):
        model.probabilities(x, entity=entity)
    assert not calls


@pytest.mark.parametrize("defect", ["nan", "negative", "above_one", "sum", "shape"])
def test_invalid_actual_predictor_results_refuse(model, monkeypatch, defect):
    x, _, _ = donor()
    values = np.full((len(x), 4), 0.25)
    if defect == "nan":
        values[0, 0] = np.nan
    elif defect == "negative":
        values[0, 0] = -0.1
    elif defect == "above_one":
        values[0, 0] = 1.1
    elif defect == "sum":
        values[0, 0] = 0.2
    else:
        values = values[:, :-1]
    monkeypatch.setattr(
        HistGradientBoostingClassifier, "predict_proba", lambda *a, **k: values
    )
    with pytest.raises(ValueError, match="PROBABILITY"):
        model.probabilities(x, entity="person")


@pytest.mark.parametrize(
    "defect",
    [
        "bytes",
        "seed",
        "source",
        "class",
        "dependency",
        "model_class",
        "model_parameters",
        "model_classes",
    ],
)
def test_model_and_metadata_changes_refuse(model, defect):
    payload = model.to_bytes()
    if defect == "bytes":
        with pytest.raises(ValueError, match="MODEL_DIGEST"):
            cat.CategoricalModel.from_trusted_bytes(
                payload + b"x", expected_sha256=cat._sha(payload)
            )
        return
    if defect.startswith("model_"):
        raw = pickle.loads(model._pickle)
        if defect == "model_class":
            raw = {"pretend": "model"}
        elif defect == "model_parameters":
            raw.random_state += 1
        else:
            raw.classes_ = np.array(["extra", *raw.classes_[1:]])
        changed = replace(model, _pickle=pickle.dumps(raw))
        payload = changed.to_bytes()
        with pytest.raises(ValueError, match="ESTIMATOR"):
            cat.CategoricalModel.from_trusted_bytes(
                payload, expected_sha256=cat._sha(payload)
            )
        return
    meta = model.metadata
    if defect == "seed":
        meta["seed"] += 1
    elif defect == "source":
        meta["source_sha256"] = "f" * 64
    elif defect == "class":
        meta["classes"][0] = "extra"
    else:
        meta["implementation"]["versions"]["scikit-learn"] = "different"
    changed = replace(model, _header=cat.codec.encode_json(meta))
    with pytest.raises(ValueError):
        changed.to_bytes()


def test_donor_weight_source_schema_and_seed_enter_training_identity(model):
    x, y, w = donor()
    cases = [
        fit(seed=102),
        fit(source_sha256="e" * 64),
        fit(weights=Weights(w.values + 1, WeightKind.DESIGN)),
        fit(features=x.rename(columns={"age": "source_age"})),
        fit(classes=CLASSES[::-1]),
    ]
    assert all(
        case.metadata["training_id"] != model.metadata["training_id"] for case in cases
    )
    assert all(case.to_bytes() != model.to_bytes() for case in cases)


def test_real_classifier_learns_feature_dependence_without_mutating_inputs():
    index = pd.Index(np.arange(24, dtype=np.int64) + 101, name="person_id")
    features = pd.DataFrame(
        {"category_predictor": np.repeat(np.arange(4, dtype=np.float64), 6)},
        index=index,
    )
    labels = pd.Series(np.repeat(CLASSES, 6), index=index)
    weights = Weights(np.ones(24), WeightKind.DESIGN)
    before_features, before_labels = features.copy(deep=True), labels.copy(deep=True)
    before_weights = weights.values.copy()
    learned = fit(
        features,
        labels,
        weights,
        config=cat.CategoricalConfig(
            max_iter=12, min_samples_leaf=1, learning_rate=0.2, max_leaf_nodes=7
        ),
    )
    recipients = pd.DataFrame(
        {"category_predictor": np.arange(4, dtype=np.float64)},
        index=pd.Index(np.arange(4, dtype=np.int64) + 501, name="person_id"),
    )
    before_recipients = recipients.copy(deep=True)
    result = learned.probabilities(recipients, entity="person")
    # Balanced weighted priors are identical. A constant-share evaluator
    # cannot produce all four different, correct maxima on these predictors.
    assert result.idxmax(axis=1).tolist() == list(CLASSES)
    assert len(np.unique(result.to_numpy(), axis=0)) == 4
    for row, label in enumerate(CLASSES):
        assert result.iloc[row][label] > result.iloc[row].drop(label).max()
    pd.testing.assert_frame_equal(features, before_features)
    pd.testing.assert_series_equal(labels, before_labels)
    pd.testing.assert_frame_equal(recipients, before_recipients)
    np.testing.assert_array_equal(weights.values, before_weights)


@pytest.mark.parametrize("boundary", ["fit_metadata_io", "prediction_unpickle"])
def test_model_consumes_exact_captured_matrix_despite_later_caller_mutation(
    monkeypatch, boundary
):
    index = pd.Index(np.arange(24, dtype=np.int64) + 101, name="person_id")
    features = pd.DataFrame(
        {"category_predictor": np.repeat(np.arange(4, dtype=np.float64), 6)},
        index=index,
    )
    labels = pd.Series(np.repeat(CLASSES, 6), index=index)
    weights = Weights(np.ones(24), WeightKind.DESIGN)
    config = cat.CategoricalConfig(
        max_iter=12, min_samples_leaf=1, learning_rate=0.2, max_leaf_nodes=7
    )
    captured = features.copy(deep=True)
    reference = fit(captured, labels, weights, config=config)
    expected = reference.probabilities(captured, entity="person")
    changed = []
    if boundary == "fit_metadata_io":
        original = cat.implementation_binding

        def mutate_then_bind():
            if not changed:
                features.iloc[:, 0] *= -1
                changed.append(True)
            return original()

        monkeypatch.setattr(cat, "implementation_binding", mutate_then_bind)
        fitted = fit(features, labels, weights, config=config)
        assert fitted.metadata["donor_matrix_sha256"] == cat._sha(
            cat.matrix_bytes(captured, entity="person")
        )
        result = fitted.probabilities(captured, entity="person")
    else:
        original = cat.pickle.loads

        def mutate_then_load(*args, **kwargs):
            if not changed:
                features.iloc[:, 0] *= -1
                features.index = features.index + 1000
                changed.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(cat.pickle, "loads", mutate_then_load)
        result = reference.probabilities(features, entity="person")
    assert changed == [True]
    assert not features.equals(captured)
    pd.testing.assert_frame_equal(result, expected)
