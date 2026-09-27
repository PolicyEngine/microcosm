"""DESIGN-weighted categorical probabilities from a real fitted classifier.

The caller owns source qualification and predictor encoding. These artifacts
are descriptive model outputs, not survey authority. The fitted-model envelope
uses pickle, like the maintained QRF artifact: load trusted producer bytes only.
A digest detects corruption; it does not authenticate an untrusted pickle.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model as weight_model
from microcosm.fit import model_input, qrf, qrf_target
from microcosm.frame import WeightKind, Weights
from microcosm.frame import weights as frame_weights
from microcosm.graph import ArtifactType, platform_fingerprint

MODEL_TYPE = ArtifactType("microcosm.fit.categorical_model", 1)
DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn")
_MAGIC = b"microcosm.fit.categorical_model/1\n"
_HEADER_LIMIT = 1024 * 1024
_MODEL_LIMIT = 512 * 1024**2


def _require(condition, reason):
    if not condition:
        raise ValueError("CATEGORICAL_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _hash(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _names(value):
    _require(
        type(value) is tuple
        and len(value) > 0
        and all(type(v) is str and v and v.strip() == v for v in value)
        and len(set(value)) == len(value),
        "NAMES",
    )
    return value


def _classes(value):
    _names(value)
    # sklearn's binary probability shape is not a one-class probability API.
    _require(len(value) >= 2, "AT_LEAST_TWO_CLASSES")
    return value


@dataclass(frozen=True)
class CategoricalConfig:
    """Explicit supported estimator controls; no hidden resampling/class weights."""

    max_iter: int = 100
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255

    def __post_init__(self):
        for name in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "max_bins"):
            _require(
                type(getattr(self, name)) is int
                and getattr(self, name)
                >= (2 if name in ("max_leaf_nodes", "max_bins") else 1),
                "CONFIG:" + name,
            )
        _require(self.max_bins <= 255, "CONFIG:max_bins")
        for name in ("learning_rate", "l2_regularization"):
            _require(
                type(getattr(self, name)) is float and np.isfinite(getattr(self, name)),
                "CONFIG:" + name,
            )
        _require(self.learning_rate > 0 and self.l2_regularization >= 0, "CONFIG_RANGE")

    def parameters(self, seed):
        _require(type(seed) is int and 0 <= seed < 2**32, "SEED")
        return dict(
            **asdict(self),
            loss="log_loss",
            max_depth=None,
            max_features=1.0,
            categorical_features=None,
            monotonic_cst=None,
            interaction_cst=None,
            warm_start=False,
            early_stopping=False,
            scoring="loss",
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-7,
            verbose=0,
            random_state=seed,
            class_weight=None,
        )


def implementation_binding():
    """Exact source/runtime identity; no cross-call source or fitted-state cache."""
    modules = (
        sys.modules[__name__],
        model_input,
        codec,
        qrf,
        qrf_target,
        weight_model,
        frame_weights,
    )
    return dict(
        platform=platform_fingerprint(),
        python=sys.version,
        versions={name: version(name) for name in DEPENDENCIES},
        sources={
            module.__name__: _sha(Path(module.__file__).read_bytes())
            for module in modules
        },
    )


def matrix_bytes(features, *, entity):
    # The maintained codec refuses cast IDs, nonfinite values and wrong dtypes.
    return model_input.encode_recipient_matrix(
        features, entity=entity, entity_ids=features.index.to_numpy(copy=True)
    )


def _metadata(metadata):
    _require(
        type(metadata) is dict
        and set(metadata)
        == {
            "protocol",
            "classes",
            "predictors",
            "entity",
            "config",
            "parameters",
            "seed",
            "weight_kind",
            "source_sha256",
            "donor_matrix_sha256",
            "labels_sha256",
            "weights_sha256",
            "rows",
            "class_weight_mass",
            "implementation",
            "training_id",
        },
        "METADATA_SCHEMA",
    )
    _require(metadata["protocol"] == MODEL_TYPE.name + "/1", "PROTOCOL")
    _require(
        type(metadata["classes"]) is list and type(metadata["predictors"]) is list,
        "SCHEMA_ROSTERS",
    )
    _classes(tuple(metadata["classes"]))
    _names(tuple(metadata["predictors"]))
    _require(
        type(metadata["entity"]) is str
        and bool(metadata["entity"])
        and "." not in metadata["entity"],
        "ENTITY",
    )
    _require(
        type(metadata["config"]) is dict
        and set(metadata["config"]) == set(CategoricalConfig.__dataclass_fields__),
        "CONFIG_SCHEMA",
    )
    config = CategoricalConfig(**metadata["config"])
    _require(
        codec.encode_json(metadata["parameters"])
        == codec.encode_json(config.parameters(metadata["seed"])),
        "PARAMETERS",
    )
    _require(metadata["weight_kind"] == WeightKind.DESIGN.value, "DESIGN_WEIGHTS")
    _require(
        all(
            _hash(metadata[k])
            for k in (
                "source_sha256",
                "donor_matrix_sha256",
                "labels_sha256",
                "weights_sha256",
                "training_id",
            )
        ),
        "METADATA_HASH",
    )
    _require(
        type(metadata["rows"]) is int and metadata["rows"] >= len(metadata["classes"]),
        "DONOR_ROWS",
    )
    mass = metadata["class_weight_mass"]
    _require(
        type(mass) is list
        and len(mass) == len(metadata["classes"])
        and all(type(v) is float and np.isfinite(v) and v > 0 for v in mass)
        and np.isfinite(sum(mass)),
        "CLASS_SUPPORT",
    )
    _require(
        codec.encode_json(metadata["implementation"])
        == codec.encode_json(implementation_binding()),
        "IMPLEMENTATION",
    )
    _require(
        metadata["training_id"]
        == _sha(
            codec.encode_json({k: v for k, v in metadata.items() if k != "training_id"})
        ),
        "TRAINING_ID",
    )


def _estimator(model, metadata):
    _require(type(model) is HistGradientBoostingClassifier, "ESTIMATOR_TYPE")
    _require(
        codec.encode_json(model.get_params(deep=False))
        == codec.encode_json(metadata["parameters"]),
        "ESTIMATOR_PARAMETERS",
    )
    labels = np.asarray(model.classes_)
    _require(
        labels.ndim == 1
        and labels.dtype.kind in "OUS"
        and all(isinstance(v, str) for v in labels.tolist()),
        "ESTIMATOR_CLASSES",
    )
    classes = tuple(labels.tolist())
    _require(
        len(classes) == len(set(classes)) == len(metadata["classes"])
        and set(classes) == set(metadata["classes"]),
        "ESTIMATOR_CLASS_SUPPORT",
    )
    _require(model.n_features_in_ == len(metadata["predictors"]), "ESTIMATOR_FEATURES")
    return classes


@dataclass(frozen=True)
class CategoricalModel:
    """Immutable serialized fitted state; no live mutable estimator escapes."""

    _header: bytes
    _pickle: bytes

    @property
    def metadata(self):
        return codec.decode_json(self._header)

    def to_bytes(self):
        _require(
            type(self._header) is bytes and type(self._pickle) is bytes, "MODEL_BYTES"
        )
        _metadata(self.metadata)
        header = codec.encode_json(
            dict(metadata=self.metadata, pickle_sha256=_sha(self._pickle))
        )
        _require(
            0 < len(header) <= _HEADER_LIMIT and 0 < len(self._pickle) <= _MODEL_LIMIT,
            "MODEL_SIZE",
        )
        return _MAGIC + len(header).to_bytes(4, "big") + header + self._pickle

    @classmethod
    def from_trusted_bytes(cls, payload, *, expected_sha256):
        """Only trusted local graph/store bytes; hashes do not make pickle safe."""
        _require(
            type(payload) is bytes
            and _hash(expected_sha256)
            and _sha(payload) == expected_sha256,
            "MODEL_DIGEST",
        )
        offset = len(_MAGIC)
        _require(
            payload.startswith(_MAGIC) and len(payload) > offset + 4, "MODEL_ENVELOPE"
        )
        length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        _require(
            0 < length <= _HEADER_LIMIT
            and offset + length < len(payload) <= offset + length + _MODEL_LIMIT,
            "MODEL_SIZE",
        )
        header = codec.decode_json(payload[offset : offset + length])
        _require(set(header) == {"metadata", "pickle_sha256"}, "MODEL_HEADER")
        _metadata(header["metadata"])
        body = payload[offset + length :]
        _require(
            _hash(header["pickle_sha256"]) and _sha(body) == header["pickle_sha256"],
            "PICKLE_DIGEST",
        )
        model = pickle.loads(body)  # noqa: S301 - trusted producer artifacts only
        _estimator(model, header["metadata"])
        return cls(codec.encode_json(header["metadata"]), body)

    def probabilities(self, features, *, entity):
        metadata = self.metadata
        _metadata(metadata)
        _require(
            entity == metadata["entity"]
            and tuple(features.columns) == tuple(metadata["predictors"]),
            "RECIPIENT_SCHEMA",
        )
        features = model_input.decode_recipient_matrix(
            matrix_bytes(features, entity=entity)
        ).features
        model = pickle.loads(self._pickle)  # noqa: S301 - retained trusted fitted state
        actual_classes = _estimator(model, metadata)
        values = model.predict_proba(features.to_numpy(copy=True))
        _require(
            type(values) is np.ndarray
            and values.dtype == np.dtype("float64")
            and values.shape == (len(features), len(actual_classes)),
            "PROBABILITY_SHAPE",
        )
        _require(
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values <= 1).all()
            and np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-12),
            "PROBABILITY_VALUES",
        )
        # Use actual fitted labels, never rely on sklearn's sorted class order.
        order = [actual_classes.index(label) for label in metadata["classes"]]
        return pd.DataFrame(
            values[:, order], index=features.index.copy(), columns=metadata["classes"]
        )


def fit_categorical(
    features, labels, *, weights, classes, entity, source_sha256, seed, config
):
    """Fit observed categorical labels with exact aligned DESIGN sample weights."""
    _classes(classes)
    _require(type(config) is CategoricalConfig, "CONFIG_TYPE")
    parameters = config.parameters(seed)
    _require(_hash(source_sha256), "SOURCE_HASH")
    matrix = matrix_bytes(features, entity=entity)
    # Fit and metadata consume exactly the captured bytes, even if later
    # implementation I/O triggers a callback that changes the caller table.
    features = model_input.decode_recipient_matrix(matrix).features
    _require(
        type(labels) is pd.Series
        and type(labels.index) is pd.Index
        and labels.index.dtype == np.dtype("int64")
        and labels.index.equals(features.index)
        and labels.index.name == features.index.name
        and len(labels) == len(features),
        "LABEL_INDEX",
    )
    values = labels.tolist()
    _require(
        all(type(v) is str for v in values) and set(values) == set(classes),
        "LABEL_CLASSES",
    )
    _require(
        type(weights) is Weights
        and weights.kind is WeightKind.DESIGN
        and len(weights) == len(features),
        "DESIGN_WEIGHTS",
    )
    w = weights.values.copy()
    _require(
        w.dtype == np.dtype("float64")
        and np.isfinite(w).all()
        and (w >= 0).all()
        and np.isfinite(w.sum())
        and w.sum() > 0,
        "WEIGHT_MASS",
    )
    mass = [
        float(w[np.array([v == label for v in values], dtype=bool)].sum())
        for label in classes
    ]
    _require(all(np.isfinite(v) and v > 0 for v in mass), "CLASS_SUPPORT")
    metadata = dict(
        protocol=MODEL_TYPE.name + "/1",
        classes=list(classes),
        predictors=list(features.columns),
        entity=entity,
        config=asdict(config),
        parameters=parameters,
        seed=seed,
        weight_kind=weights.kind.value,
        source_sha256=source_sha256,
        donor_matrix_sha256=_sha(matrix),
        labels_sha256=_sha(codec.encode_json(dict(labels=values))),
        weights_sha256=_sha(w.astype("<f8").tobytes()),
        rows=len(features),
        class_weight_mass=mass,
        implementation=implementation_binding(),
    )
    metadata["training_id"] = _sha(codec.encode_json(metadata))
    model = HistGradientBoostingClassifier(**parameters)
    model.fit(features.to_numpy(copy=True), np.array(values), sample_weight=w)
    _estimator(model, metadata)
    result = CategoricalModel(
        codec.encode_json(metadata),
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
    )
    # Validate actual model and canonical envelope before returning its artifact.
    payload = result.to_bytes()
    return CategoricalModel.from_trusted_bytes(payload, expected_sha256=_sha(payload))
