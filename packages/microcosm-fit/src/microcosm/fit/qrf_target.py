"""Reusable target forests with the exact legacy sequential QRF draw protocol.

Training advances a donor-only checkpoint independently of every recipient.
Application consumes a normal :class:`QRFChainState` and the *raw*, ordered
float64 recipient prefix, delegating to the existing gated draw implementation.
This is not the stateless/keyed-uniform graph QRF protocol. Callers must retain
the raw prefix as an immutable dependency; values alone cannot prove that a
caller has not snapped or otherwise transformed them.

Artifacts are PLATFORM_BITWISE and bind the graph platform fingerprint, exact
runtime versions, and implementation source hashes. Their envelope is checked
before unpickling. An expected content SHA detects corruption; it does NOT make
an untrusted pickle safe. Only load artifacts from trusted local producers or
an authenticated content store. The private fitted estimator is owned by one
synchronous apply at a time (the existing predictor swaps its worker setting).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from microcosm.fit import model as weight_resolution
from microcosm.fit import qrf
from microcosm.fit.model import DESIGN_WEIGHTS, WeightSpec
from microcosm.frame import Frame
from microcosm.graph import ArtifactType, Numeric, platform_fingerprint

LEGACY_QRF_TARGET_TYPE = ArtifactType("microcosm.fit.qrf.legacy_target", 1)
_MAGIC = b"microcosm.fit.qrf.legacy_target/1\n"
_REGIMES = {
    qrf.Regime.THREE_SIGN,
    qrf.Regime.ZERO_INFLATED_POSITIVE,
    qrf.Regime.ZERO_INFLATED_NEGATIVE,
    qrf.Regime.SIGN_ONLY,
    qrf.Regime.POSITIVE_ONLY,
    qrf.Regime.NEGATIVE_ONLY,
    qrf.Regime.DEGENERATE_ZERO,
}
_TRAINING_FIELDS = {
    "schema_version",
    "predictors",
    "targets",
    "completed_targets",
    "entity",
    "weight_kind",
    "weight_sha256",
    "model_config",
    "donor_index",
    "fit_rng_state",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _implementation_binding() -> dict[str, object]:
    """Conservative same-implementation, same-platform cache/load boundary."""
    return {
        "numeric": Numeric.PLATFORM_BITWISE.value,
        "platform": platform_fingerprint(),
        "python": sys.version,
        "versions": {
            package: version(package)
            for package in (
                "microcosm-fit",
                "microcosm-frame",
                "microcosm-graph",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "quantile-forest",
            )
        },
        "qrf_sha256": _sha(Path(qrf.__file__).read_bytes()),
        "weight_resolution_sha256": _sha(Path(weight_resolution.__file__).read_bytes()),
        "target_protocol_sha256": _sha(Path(__file__).read_bytes()),
    }


@dataclass(frozen=True)
class LegacyQRFTrainingState:
    """Immutable donor/config/order/fit-RNG state, with no recipient/draw fields.

    Use ``from_chain`` once at chain initialization; subsequent training targets
    consume ``artifact.next_training_state`` and never depend on apply output.
    ``to_dict``/``from_dict`` support JSON checkpoints between local workers.
    """

    _canonical_json: str

    def __post_init__(self) -> None:
        value = json.loads(self._canonical_json)
        if not isinstance(value, dict) or set(value) != _TRAINING_FIELDS:
            raise ValueError("Invalid legacy QRF training state fields.")
        if _canonical(value) != self._canonical_json:
            raise ValueError("Legacy QRF training state must use canonical JSON.")
        self._chain()

    def _chain(self) -> qrf.QRFChainState:
        # Reuse the authoritative legacy validators. This fixed neutral draw
        # state exists only inside validation, never in training serialization
        # or identity, and never reaches application.
        return qrf.QRFChainState.from_dict(
            {
                **self.to_dict(),
                "recipient_index": None,
                "draw_rng_state": np.random.PCG64(0).state,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._canonical_json)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LegacyQRFTrainingState:
        return cls(_canonical(dict(value)))

    @classmethod
    def from_chain(cls, state: qrf.QRFChainState) -> LegacyQRFTrainingState:
        if not isinstance(state, qrf.QRFChainState):
            raise TypeError("state must be a QRFChainState.")
        value = state.to_dict()
        del value["recipient_index"]
        del value["draw_rng_state"]
        return cls.from_dict(value)

    @property
    def is_complete(self) -> bool:
        return self._chain().is_complete

    @property
    def next_target(self) -> str | None:
        return self._chain().next_target


def _training_id(state, donor_sha256, implementation) -> str:
    return _sha(
        _canonical(
            {
                "type": LEGACY_QRF_TARGET_TYPE.name,
                "schema_version": LEGACY_QRF_TARGET_TYPE.schema_version,
                "training_state": state.to_dict(),
                "donor_sha256": donor_sha256,
                "implementation": implementation,
            }
        ).encode()
    )


def _consumed_values_sha256(table: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """Hash the exact float64 estimator inputs without a whole-table copy."""
    digest = hashlib.sha256(_canonical([list(columns), len(table)]).encode())
    for column in columns:
        values = np.ascontiguousarray(table[column].to_numpy(dtype="<f8"))
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _validate_training_metadata(
    before, after, donor_sha256, training_id, implementation
):
    if implementation != _implementation_binding():
        raise ValueError("Legacy QRF artifact implementation/platform mismatch.")
    if not _is_sha(donor_sha256) or training_id != _training_id(
        before, donor_sha256, implementation
    ):
        raise ValueError("Legacy QRF artifact training identity mismatch.")
    initial = before._chain()
    successor = after._chain()
    if initial.is_complete:
        raise ValueError("Model artifact training state is already complete.")
    expected = replace(
        initial,
        completed_targets=(*initial.completed_targets, initial.next_target),
        fit_rng_state_json=successor.fit_rng_state_json,
    )
    if successor != expected:
        raise ValueError("Legacy QRF artifact next training state mismatch.")


@dataclass(frozen=True)
class LegacyQRFTargetArtifact:
    """One fitted target plus a donor-only training successor and provenance.

    ``training_id`` is a training dependency identity, not a serialized model
    digest. For transport/cache integrity hash the actual ``to_bytes()`` output.
    No recipient identity or draw RNG occurs in either identity or payload.
    """

    training_state: LegacyQRFTrainingState
    next_training_state: LegacyQRFTrainingState
    donor_sha256: str
    training_id: str
    _implementation_json: str
    _target_model: qrf._TargetModel
    numeric: ClassVar[Numeric] = Numeric.PLATFORM_BITWISE

    @property
    def target(self) -> str:
        target = self.training_state.next_target
        if target is None:
            raise ValueError("Model artifact training state is already complete.")
        return target

    @property
    def regime(self) -> str:
        return self._target_model.regime

    @property
    def platform(self) -> str:
        return json.loads(self._implementation_json)["platform"]

    def _validate(self) -> None:
        implementation = json.loads(self._implementation_json)
        _validate_training_metadata(
            self.training_state,
            self.next_training_state,
            self.donor_sha256,
            self.training_id,
            implementation,
        )
        before = self.training_state._chain()
        columns = (*before.predictors, *before.completed_targets)
        if (
            type(self._target_model) is not qrf._TargetModel
            or self._target_model.columns != columns
            or self.regime not in _REGIMES
        ):
            raise ValueError("Legacy QRF fitted model does not match its metadata.")

    def to_bytes(self) -> bytes:
        """Encode a trusted-local versioned envelope without mutating the model."""
        self._validate()
        payload = pickle.dumps(self._target_model, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "schema_version": 1,
            "type": LEGACY_QRF_TARGET_TYPE.name,
            "training_state": self.training_state.to_dict(),
            "next_training_state": self.next_training_state.to_dict(),
            "donor_sha256": self.donor_sha256,
            "training_id": self.training_id,
            "implementation": json.loads(self._implementation_json),
            "target": self.target,
            "regime": self.regime,
            "pickle_sha256": _sha(payload),
        }
        header = _canonical(metadata).encode()
        return _MAGIC + len(header).to_bytes(8, "big") + header + payload

    @classmethod
    def from_trusted_bytes(
        cls, data: bytes, *, expected_sha256: str
    ) -> LegacyQRFTargetArtifact:
        """Load only trusted producer bytes; a hash is not pickle authentication."""
        if not _is_sha(expected_sha256) or _sha(data) != expected_sha256:
            raise ValueError("Legacy QRF artifact content digest mismatch.")
        offset = len(_MAGIC)
        if not data.startswith(_MAGIC) or len(data) < offset + 8:
            raise ValueError("Invalid legacy QRF model envelope.")
        size = int.from_bytes(data[offset : offset + 8], "big")
        offset += 8
        if not 0 < size < len(data) - offset:
            raise ValueError("Invalid legacy QRF model header length.")
        metadata = json.loads(data[offset : offset + size])
        expected_fields = {
            "schema_version",
            "type",
            "training_state",
            "next_training_state",
            "donor_sha256",
            "training_id",
            "implementation",
            "target",
            "regime",
            "pickle_sha256",
        }
        if (
            not isinstance(metadata, dict)
            or set(metadata) != expected_fields
            or type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != 1
            or metadata["type"] != LEGACY_QRF_TARGET_TYPE.name
        ):
            raise ValueError("Invalid legacy QRF model metadata/schema version.")
        if metadata["implementation"] != _implementation_binding():
            raise ValueError("Legacy QRF artifact implementation/platform mismatch.")
        payload = data[offset + size :]
        if _sha(payload) != metadata["pickle_sha256"]:
            raise ValueError("Legacy QRF model pickle digest mismatch.")
        before = LegacyQRFTrainingState.from_dict(metadata["training_state"])
        after = LegacyQRFTrainingState.from_dict(metadata["next_training_state"])
        _validate_training_metadata(
            before,
            after,
            metadata["donor_sha256"],
            metadata["training_id"],
            metadata["implementation"],
        )
        if (
            metadata["target"] != before.next_target
            or metadata["regime"] not in _REGIMES
        ):
            raise ValueError("Legacy QRF artifact training identity mismatch.")
        # Integrity, schema, implementation, and platform were checked above.
        # This is still arbitrary-code-capable pickle: trust is a caller duty.
        model = pickle.loads(payload)  # noqa: S301 - trusted local artifacts only
        artifact = cls(
            before,
            after,
            metadata["donor_sha256"],
            metadata["training_id"],
            _canonical(metadata["implementation"]),
            model,
        )
        artifact._validate()
        if artifact.regime != metadata["regime"]:
            raise ValueError("Legacy QRF model regime metadata mismatch.")
        return artifact


def fit_target(
    model: qrf.RegimeGatedQRF,
    donors: Frame | pd.DataFrame,
    *,
    state: LegacyQRFTrainingState,
    weights: WeightSpec = DESIGN_WEIGHTS,
) -> LegacyQRFTargetArtifact:
    """Fit exactly the next legacy target using observed donor prior targets."""
    if not isinstance(state, LegacyQRFTrainingState):
        raise TypeError("state must be a LegacyQRFTrainingState.")
    before = state._chain()
    model._validate_chain_config(before)
    if before.is_complete:
        raise ValueError("Legacy QRF training chain is complete.")
    resolved = qrf._resolve_qrf_fit_input(
        donors, list(before.predictors), list(before.targets), weights
    )
    model._validate_chain_donor(before, resolved)
    columns = (*before.predictors, *before.completed_targets)
    target = before.next_target
    donor_sha256 = _consumed_values_sha256(resolved.table, (*columns, target))
    implementation = _implementation_binding()
    rng = qrf._rng_from_state_json(before.fit_rng_state_json, stream="fit")
    fitted = model._fit_target(
        features=resolved.table.loc[:, list(columns)].to_numpy(dtype=np.float64),
        y=resolved.table[target].to_numpy(dtype=np.float64),
        columns=columns,
        weights=resolved.weights,
        rng=rng,
    )
    after = replace(
        before,
        completed_targets=(*before.completed_targets, target),
        fit_rng_state_json=qrf._rng_state_json(rng),
    )
    return LegacyQRFTargetArtifact(
        state,
        LegacyQRFTrainingState.from_chain(after),
        donor_sha256,
        _training_id(state, donor_sha256, implementation),
        _canonical(implementation),
        fitted,
    )


def apply_target(
    artifact: LegacyQRFTargetArtifact,
    recipient_predictors: Frame | pd.DataFrame,
    raw_prior_draws: pd.DataFrame,
    *,
    state: qrf.QRFChainState,
) -> qrf.QRFChainStepResult:
    """Draw one fitted target, preserving raw conditioning and legacy RNG order."""
    if not isinstance(artifact, LegacyQRFTargetArtifact):
        raise TypeError("artifact must be a LegacyQRFTargetArtifact.")
    if not isinstance(state, qrf.QRFChainState):
        raise TypeError("state must be a QRFChainState.")
    if state.is_complete:
        raise ValueError("Legacy QRF application chain is complete.")
    artifact._validate()
    if LegacyQRFTrainingState.from_chain(state) != artifact.training_state:
        raise ValueError("Legacy QRF artifact/application training state mismatch.")
    # Match fit_draw_next's explicit configured worker-count guard on resume.
    if state.fit_n_jobs != qrf._fit_n_jobs():
        raise ValueError("QRF chain model configuration changed: fit worker count.")
    recipient = qrf.RegimeGatedQRF._chain_recipient_table(recipient_predictors, state)
    identity = qrf._index_identity(recipient.index)
    if state.recipient_index is not None and state.recipient_index != identity:
        raise ValueError("QRF chain recipient index/order changed since first draw.")
    qrf.RegimeGatedQRF._validate_raw_prior_draws(
        raw_prior_draws, recipient.index, state
    )
    augmented = recipient.loc[:, list(state.predictors)].copy()
    for prior in state.completed_targets:
        augmented[prior] = raw_prior_draws[prior].to_numpy(copy=False)
    rng = qrf._rng_from_state_json(state.draw_rng_state_json, stream="draw")
    raw = np.asarray(
        qrf._draw_target_with_rng(augmented, artifact._target_model, rng),
        dtype=np.float64,
    )
    raw.setflags(write=False)
    advanced = replace(
        state,
        completed_targets=(*state.completed_targets, artifact.target),
        recipient_index=identity,
        fit_rng_state_json=artifact.next_training_state._chain().fit_rng_state_json,
        draw_rng_state_json=qrf._rng_state_json(rng),
    )
    return qrf.QRFChainStepResult(
        target=artifact.target,
        raw_draw=raw,
        state=advanced,
        regime=artifact.regime,
        weight_kind=state.weight_kind,
    )
