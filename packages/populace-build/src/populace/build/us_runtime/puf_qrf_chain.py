"""Fresh-process checkpoints for the primary US PUF QRF target chain."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from populace.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from populace.build.stage_profile import profile_stage
from populace.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    PufTaxDetailChainInputs,
    finalize_us_puf_tax_detail_predictions,
    prepare_us_puf_tax_detail_chain_inputs,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.fit import QRF, QRFChainState
from populace.frame import EntitySchema, Frame, WeightKind, Weights

# v2 (populace#515): the checkpointed donor frame now carries the E19200 ->
# mortgage-only concept carve (US_PUF_E19200_HOME_MORTGAGE_SHARE). Loading
# validates only schema/digest/kind/role -- not donor construction identity --
# so a v1 checkpoint initialized pre-carve would keep fitting and drawing the
# uncarved levels under carved code. The bump rejects every pre-carve
# checkpoint and forces re-initialization through the carved constructor.
PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION = 2
PRIMARY_QRF_MANIFEST_FILENAME = "manifest.json"
PRIMARY_QRF_DONOR_FILENAME = "donor.frame.h5"
PRIMARY_QRF_RECIPIENT_FILENAME = "recipient.frame.h5"
PRIMARY_QRF_TARGETS_DIRNAME = "targets"
PRIMARY_QRF_TARGET_ORDER = (
    *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
PRIMARY_QRF_TARGET_ORDER_SHA256 = (
    "556d713d029840848aba548d9c991842a54ea3cdbdea650c0be57d5ec5782661"
)

_ARTIFACT_KIND = "populace_primary_puf_qrf_chain"
_RAW_TARGET_ARTIFACT_KIND = "populace_primary_puf_qrf_raw_target"
_METADATA_DATASET = "metadata_json"
_RAW_DRAW_BITS_DATASET = "raw_draw_bits"

__all__ = [
    "PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION",
    "PRIMARY_QRF_TARGET_ORDER",
    "PRIMARY_QRF_TARGET_ORDER_SHA256",
    "finalize_primary_puf_qrf_chain",
    "initialize_primary_puf_qrf_chain",
    "load_primary_puf_qrf_predictions",
    "run_primary_puf_qrf_chain",
    "run_primary_puf_qrf_target",
]


def initialize_primary_puf_qrf_chain(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    checkpoint_dir: str | Path,
    *,
    predictors: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    seed: int = 0,
    n_estimators: int = 100,
) -> dict[str, object]:
    """Write immutable donor/recipient inputs and the initial RNG manifest."""

    root = Path(checkpoint_dir)
    manifest_path = root / PRIMARY_QRF_MANIFEST_FILENAME
    if manifest_path.exists():
        raise FileExistsError(f"Primary QRF manifest already exists: {manifest_path}")
    root.mkdir(parents=True, exist_ok=True)
    (root / PRIMARY_QRF_TARGETS_DIRNAME).mkdir(exist_ok=True)

    inputs = prepare_us_puf_tax_detail_chain_inputs(
        frame,
        donor_tax_units,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
    )
    target_order = inputs.target_order
    if (
        tuple(person_outputs) == PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
        and tuple(tax_unit_outputs) == PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
        and _ordered_strings_sha256(target_order) != PRIMARY_QRF_TARGET_ORDER_SHA256
    ):
        raise AssertionError("The production primary QRF target order changed.")

    donor_path = root / PRIMARY_QRF_DONOR_FILENAME
    write_frame_checkpoint(
        donor_path,
        inputs.donor_frame,
        metadata={
            "artifact_kind": _ARTIFACT_KIND,
            "role": "donor",
            "target_order": list(target_order),
        },
    )
    recipient_frame, identity_columns = _recipient_checkpoint_frame(frame, inputs)
    recipient_path = root / PRIMARY_QRF_RECIPIENT_FILENAME
    write_frame_checkpoint(
        recipient_path,
        recipient_frame,
        metadata={
            "artifact_kind": _ARTIFACT_KIND,
            "identity_columns": list(identity_columns),
            "predictors": list(inputs.predictors),
            "role": "recipient",
            "target_order": list(target_order),
        },
    )

    model = QRF(n_estimators=n_estimators, seed=seed)
    state = model.start_chain(
        inputs.donor_frame,
        list(inputs.predictors),
        list(target_order),
        weights="design",
    )
    recipient_table = recipient_frame.table("tax_unit")
    manifest: dict[str, object] = {
        "artifact_kind": _ARTIFACT_KIND,
        "donor_checkpoint_sha256": _file_sha256(donor_path),
        "donor_filename": PRIMARY_QRF_DONOR_FILENAME,
        "initial_state": state.to_dict(),
        "n_estimators": int(n_estimators),
        "person_outputs": list(inputs.person_outputs),
        "predictors": list(inputs.predictors),
        "production_target_order": target_order == PRIMARY_QRF_TARGET_ORDER,
        "recipient_checkpoint_sha256": _file_sha256(recipient_path),
        "recipient_filename": PRIMARY_QRF_RECIPIENT_FILENAME,
        "recipient_identity_columns": list(identity_columns),
        "recipient_identity_sha256": _recipient_identity_sha256(
            recipient_table, identity_columns
        ),
        "recipient_rows": len(recipient_table),
        "schema_version": PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
        "seed": int(seed),
        "target_order": list(target_order),
        "target_order_sha256": _ordered_strings_sha256(target_order),
        "tax_unit_outputs": list(inputs.tax_unit_outputs),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def run_primary_puf_qrf_chain(
    checkpoint_dir: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Run or resume one fresh interpreter per target in exact chain order."""

    root = Path(checkpoint_dir).resolve()
    manifest = _load_manifest(root)
    target_order = _manifest_strings(manifest, "target_order")
    initial_state = manifest.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("Primary QRF manifest initial_state must be an object.")
    state = QRFChainState.from_dict(initial_state)
    existing = [
        _target_path(root, manifest, index).exists()
        for index in range(len(target_order))
    ]
    if False in existing:
        first_gap = existing.index(False)
        later = [
            index for index in range(first_gap + 1, len(existing)) if existing[index]
        ]
        if later:
            raise ValueError(
                "Primary QRF checkpoints have a non-contiguous prefix: "
                f"target {first_gap} is missing but later target(s) {later} exist."
            )
    child_environment = None
    if environment is not None:
        child_environment = {**os.environ, **dict(environment)}
    for target_index, _target in enumerate(target_order):
        target_path = _target_path(root, manifest, target_index)
        if target_path.exists():
            _raw_draw, state = _load_target_checkpoint(
                root, manifest, target_index, expected_state=state
            )
            continue
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "populace.build.us_runtime.puf_qrf_worker",
                "--checkpoint-dir",
                str(root),
                "--target-index",
                str(target_index),
            ],
            env=child_environment,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Primary PUF QRF target {target_index} ({target_order[target_index]}) "
                f"failed with exit code {completed.returncode}."
            )
        _raw_draw, state = _load_target_checkpoint(
            root, manifest, target_index, expected_state=state
        )


def run_primary_puf_qrf_target(
    checkpoint_dir: str | Path,
    target_index: int,
) -> Path:
    """Fit and draw exactly one target, then atomically checkpoint raw bits."""

    root = Path(checkpoint_dir).resolve()
    manifest = _load_manifest(root)
    target_order = _manifest_strings(manifest, "target_order")
    if target_index < 0 or target_index >= len(target_order):
        raise ValueError(
            f"target_index must be in [0, {len(target_order)}), got {target_index}."
        )
    target_path = _target_path(root, manifest, target_index)
    if target_path.exists():
        _load_target_checkpoint(root, manifest, target_index)
        return target_path

    with profile_stage(
        f"primary_qrf_{target_index:03d}_{target_order[target_index]}", root.parent
    ):
        donor = _load_bound_frame(
            root,
            manifest,
            filename_key="donor_filename",
            digest_key="donor_checkpoint_sha256",
            role="donor",
        )
        recipient = _load_bound_frame(
            root,
            manifest,
            filename_key="recipient_filename",
            digest_key="recipient_checkpoint_sha256",
            role="recipient",
        )
        recipient_table = recipient.table("tax_unit")
        identity_columns = _manifest_strings(manifest, "recipient_identity_columns")
        actual_identity = _recipient_identity_sha256(recipient_table, identity_columns)
        if actual_identity != manifest.get("recipient_identity_sha256"):
            raise ValueError("Primary QRF recipient identity or row order changed.")

        raw_prior = pd.DataFrame(index=recipient_table.index)
        state_payload = manifest.get("initial_state")
        if not isinstance(state_payload, dict):
            raise ValueError("Primary QRF manifest initial_state must be an object.")
        state = QRFChainState.from_dict(state_payload)
        for prior_index in range(target_index):
            prior_draw, state = _load_target_checkpoint(
                root, manifest, prior_index, expected_state=state
            )
            raw_prior[target_order[prior_index]] = prior_draw

        model = QRF(
            n_estimators=_manifest_integer(manifest, "n_estimators"),
            seed=_manifest_integer(manifest, "seed"),
        )
        predictors = _manifest_strings(manifest, "predictors")
        recipient_predictors = recipient_table.loc[:, list(predictors)]
        result = model.fit_draw_next(
            donor,
            recipient_predictors,
            raw_prior,
            state=state,
            weights="design",
        )
        if result.target != target_order[target_index]:
            raise AssertionError(
                f"QRF step returned {result.target!r}, expected "
                f"{target_order[target_index]!r}."
            )
        _write_target_checkpoint(
            target_path,
            manifest=manifest,
            target_index=target_index,
            target=result.target,
            raw_draw=result.raw_draw,
            state_before=state,
            state_after=result.state,
            regime=result.regime,
            weight_kind=result.weight_kind,
        )
    return target_path


def load_primary_puf_qrf_predictions(
    checkpoint_dir: str | Path,
) -> pd.DataFrame:
    """Load the complete raw target chain after validating every checkpoint."""

    root = Path(checkpoint_dir).resolve()
    manifest = _load_manifest(root)
    recipient = _load_bound_frame(
        root,
        manifest,
        filename_key="recipient_filename",
        digest_key="recipient_checkpoint_sha256",
        role="recipient",
    )
    target_order = _manifest_strings(manifest, "target_order")
    predictions = pd.DataFrame(index=recipient.table("tax_unit").index)
    state_payload = manifest.get("initial_state")
    if not isinstance(state_payload, dict):
        raise ValueError("Primary QRF manifest initial_state must be an object.")
    state = QRFChainState.from_dict(state_payload)
    for target_index, target in enumerate(target_order):
        raw_draw, state = _load_target_checkpoint(
            root, manifest, target_index, expected_state=state
        )
        predictions[target] = raw_draw
    if not state.is_complete:
        raise ValueError("Primary QRF target checkpoints do not complete the chain.")
    return predictions


def finalize_primary_puf_qrf_chain(
    frame: Frame,
    checkpoint_dir: str | Path,
    *,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
) -> tuple[Frame, str]:
    """Finalize all raw checkpoints onto ``frame`` and return fit weight kind."""

    root = Path(checkpoint_dir).resolve()
    manifest = _load_manifest(root)
    _assert_live_recipient_identity(frame, manifest)
    donor_frame = _load_bound_frame(
        root,
        manifest,
        filename_key="donor_filename",
        digest_key="donor_checkpoint_sha256",
        role="donor",
    )
    donor = donor_frame.table("tax_unit").drop(columns=["tax_unit_id"]).copy()
    donor["weight"] = donor_frame.weights_for("tax_unit").values
    predictions = load_primary_puf_qrf_predictions(root)
    finalized = finalize_us_puf_tax_detail_predictions(
        frame,
        donor,
        predictions,
        person_outputs=_manifest_strings(manifest, "person_outputs"),
        tax_unit_outputs=_manifest_strings(manifest, "tax_unit_outputs"),
        tail_bound_diagnostics=tail_bound_diagnostics,
    )
    initial_state = manifest.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("Primary QRF manifest initial_state must be an object.")
    weight_kind = QRFChainState.from_dict(initial_state).weight_kind
    return finalized, weight_kind


def _recipient_checkpoint_frame(
    frame: Frame,
    inputs: PufTaxDetailChainInputs,
) -> tuple[Frame, tuple[str, ...]]:
    tax_unit = frame.table("tax_unit")
    selected = tax_unit.index.isin(inputs.recipient_features.index)
    selected_tax_units = tax_unit.loc[selected]
    if not selected_tax_units.index.equals(inputs.recipient_features.index):
        raise ValueError("PUF recipient feature extraction changed tax-unit row order.")

    primary_id = frame.schema.entity_id_column("tax_unit")
    identity_candidates = (
        primary_id,
        support_source_id_column("tax_unit"),
        support_channel_column("tax_unit"),
        support_clone_index_column("tax_unit"),
    )
    identity_columns = tuple(
        column for column in identity_candidates if column in selected_tax_units
    )
    recipient = selected_tax_units.loc[:, list(identity_columns)].copy()
    for predictor in inputs.predictors:
        recipient[predictor] = inputs.recipient_features[predictor]
    if not np.array_equal(
        recipient[primary_id].to_numpy(), inputs.recipient_tax_unit_ids
    ):
        raise ValueError("PUF recipient tax-unit IDs changed during preparation.")

    person = pd.DataFrame(
        {
            "person_id": np.arange(1, len(recipient) + 1, dtype=np.int64),
            "person_tax_unit_id": recipient[primary_id].to_numpy(copy=True),
        }
    )
    schema = EntitySchema(group_entities=("tax_unit",))
    return (
        Frame(
            {"person": person, "tax_unit": recipient},
            schema,
            {
                "tax_unit": Weights(
                    np.ones(len(recipient), dtype=np.float64),
                    WeightKind.DESIGN,
                )
            },
        ),
        identity_columns,
    )


def _load_bound_frame(
    root: Path,
    manifest: Mapping[str, object],
    *,
    filename_key: str,
    digest_key: str,
    role: str,
) -> Frame:
    filename = manifest.get(filename_key)
    expected_digest = manifest.get(digest_key)
    if not isinstance(filename, str) or not isinstance(expected_digest, str):
        raise ValueError(f"Primary QRF manifest has invalid {role} binding.")
    path = root / filename
    actual_digest = _file_sha256(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"Primary QRF {role} checkpoint digest changed: expected "
            f"{expected_digest}, got {actual_digest}."
        )
    loaded = load_frame_checkpoint(path)
    if loaded.metadata.get("artifact_kind") != _ARTIFACT_KIND:
        raise ValueError(f"Primary QRF {role} checkpoint has the wrong artifact kind.")
    if loaded.metadata.get("role") != role:
        raise ValueError(f"Primary QRF checkpoint role is not {role!r}.")
    return loaded.frame


def _write_target_checkpoint(
    path: Path,
    *,
    manifest: Mapping[str, object],
    target_index: int,
    target: str,
    raw_draw: np.ndarray,
    state_before: QRFChainState,
    state_after: QRFChainState,
    regime: str,
    weight_kind: str,
) -> None:
    draw = np.ascontiguousarray(raw_draw, dtype="<f8")
    metadata = {
        "artifact_kind": _RAW_TARGET_ARTIFACT_KIND,
        "manifest_sha256": _mapping_sha256(manifest),
        "raw_draw_sha256": hashlib.sha256(draw.view("<u8").tobytes()).hexdigest(),
        "recipient_identity_sha256": manifest["recipient_identity_sha256"],
        "regime": regime,
        "schema_version": PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
        "state_after": state_after.to_dict(),
        "state_before_sha256": _mapping_sha256(state_before.to_dict()),
        "target": target,
        "target_index": target_index,
        "weight_kind": weight_kind,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, mode="w") as h5:
            h5.create_dataset(
                _METADATA_DATASET,
                data=np.frombuffer(_canonical_json(metadata).encode(), dtype=np.uint8),
                track_times=False,
            )
            h5.create_dataset(
                _RAW_DRAW_BITS_DATASET,
                data=draw.view("<u8"),
                dtype="<u8",
                track_times=False,
            )
            h5.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_target_checkpoint(
    root: Path,
    manifest: Mapping[str, object],
    target_index: int,
    *,
    expected_state: QRFChainState | None = None,
) -> tuple[np.ndarray, QRFChainState]:
    target_order = _manifest_strings(manifest, "target_order")
    path = _target_path(root, manifest, target_index)
    if not path.is_file():
        raise FileNotFoundError(f"Missing primary QRF target checkpoint: {path}")
    with h5py.File(path, mode="r") as h5:
        if _METADATA_DATASET not in h5 or _RAW_DRAW_BITS_DATASET not in h5:
            raise ValueError(f"Malformed primary QRF target checkpoint: {path}")
        metadata = json.loads(bytes(h5[_METADATA_DATASET][...]).decode())
        bits = np.asarray(h5[_RAW_DRAW_BITS_DATASET], dtype="<u8")
    if not isinstance(metadata, dict):
        raise ValueError(f"Primary QRF target metadata must be an object: {path}")
    expected = {
        "artifact_kind": _RAW_TARGET_ARTIFACT_KIND,
        "manifest_sha256": _mapping_sha256(manifest),
        "recipient_identity_sha256": manifest["recipient_identity_sha256"],
        "target": target_order[target_index],
        "target_index": target_index,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Primary QRF target checkpoint {path} has invalid {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}."
            )
    if len(bits) != _manifest_integer(manifest, "recipient_rows"):
        raise ValueError(f"Primary QRF target checkpoint {path} has wrong row count.")
    actual_draw_digest = hashlib.sha256(bits.tobytes()).hexdigest()
    if metadata.get("raw_draw_sha256") != actual_draw_digest:
        raise ValueError(
            f"Primary QRF target checkpoint {path} raw draw digest is invalid."
        )
    if expected_state is not None and metadata.get(
        "state_before_sha256"
    ) != _mapping_sha256(expected_state.to_dict()):
        raise ValueError(
            f"Primary QRF target checkpoint {path} does not continue the prior RNG state."
        )
    state_after = metadata.get("state_after")
    if not isinstance(state_after, dict):
        raise ValueError(f"Primary QRF target checkpoint {path} lacks state_after.")
    state = QRFChainState.from_dict(state_after)
    if state.completed_targets != tuple(target_order[: target_index + 1]):
        raise ValueError(
            f"Primary QRF target checkpoint {path} has a non-contiguous target prefix."
        )
    return bits.view("<f8").astype(np.float64, copy=False), state


def _target_path(
    root: Path,
    manifest: Mapping[str, object],
    target_index: int,
) -> Path:
    targets = _manifest_strings(manifest, "target_order")
    if target_index < 0 or target_index >= len(targets):
        raise ValueError(f"Invalid primary QRF target index {target_index}.")
    safe_target = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in targets[target_index]
    )
    return root / PRIMARY_QRF_TARGETS_DIRNAME / f"{target_index:03d}__{safe_target}.h5"


def _load_manifest(root: Path) -> dict[str, object]:
    path = root / PRIMARY_QRF_MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid primary QRF manifest JSON: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Primary QRF manifest must contain an object: {path}")
    if manifest.get("artifact_kind") != _ARTIFACT_KIND:
        raise ValueError("Primary QRF manifest has the wrong artifact kind.")
    if manifest.get("schema_version") != PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported primary QRF checkpoint schema version.")
    target_order = _manifest_strings(manifest, "target_order")
    if manifest.get("target_order_sha256") != _ordered_strings_sha256(target_order):
        raise ValueError("Primary QRF manifest target order digest is invalid.")
    initial_state = manifest.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("Primary QRF manifest initial_state must be an object.")
    state = QRFChainState.from_dict(initial_state)
    if state.targets != target_order or state.completed_targets:
        raise ValueError("Primary QRF initial state has invalid target order/prefix.")
    predictors = _manifest_strings(manifest, "predictors")
    person_outputs = _manifest_strings(manifest, "person_outputs")
    tax_unit_outputs = _manifest_strings(manifest, "tax_unit_outputs")
    if state.predictors != predictors:
        raise ValueError("Primary QRF manifest predictors disagree with initial state.")
    if (*person_outputs, *tax_unit_outputs) != target_order:
        raise ValueError(
            "Primary QRF person/tax-unit output partition does not match target order."
        )
    if state.n_estimators != _manifest_integer(
        manifest, "n_estimators"
    ) or state.seed != _manifest_integer(manifest, "seed"):
        raise ValueError(
            "Primary QRF manifest model config disagrees with initial state."
        )
    production_target_order = manifest.get("production_target_order")
    if not isinstance(production_target_order, bool):
        raise ValueError("Primary QRF production_target_order must be boolean.")
    if production_target_order and target_order != PRIMARY_QRF_TARGET_ORDER:
        raise ValueError("Primary QRF production target order changed.")
    return manifest


def _assert_live_recipient_identity(
    frame: Frame,
    manifest: Mapping[str, object],
) -> None:
    identity_columns = _manifest_strings(manifest, "recipient_identity_columns")
    tax_unit = frame.table("tax_unit")
    channel = support_channel_column("tax_unit")
    if channel not in tax_unit:
        raise ValueError("Live finalization frame lacks PUF support-channel identity.")
    missing_identity = [column for column in identity_columns if column not in tax_unit]
    if missing_identity:
        raise ValueError(
            f"Live finalization frame lacks identity columns: {missing_identity}."
        )
    live = tax_unit.loc[
        tax_unit[channel].to_numpy() == PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        list(identity_columns),
    ]
    expected_rows = _manifest_integer(manifest, "recipient_rows")
    expected_digest = manifest.get("recipient_identity_sha256")
    if len(live) != expected_rows or not isinstance(expected_digest, str):
        raise ValueError(
            "Live finalization frame has a different PUF recipient surface."
        )
    if _recipient_identity_sha256(live, identity_columns) != expected_digest:
        raise ValueError(
            "Live finalization frame changed PUF recipient identity or row order."
        )


def _manifest_strings(manifest: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Primary QRF manifest {key!r} must be a string list.")
    return tuple(value)


def _manifest_integer(manifest: Mapping[str, object], key: str) -> int:
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Primary QRF manifest {key!r} must be an integer.")
    return value


def _recipient_identity_sha256(
    table: pd.DataFrame,
    identity_columns: Sequence[str],
) -> str:
    missing = [column for column in identity_columns if column not in table]
    if missing:
        raise ValueError(f"Recipient identity columns missing: {missing}.")
    identity = table.loc[:, list(identity_columns)]
    header = {
        "columns": list(identity_columns),
        "dtypes": [str(identity[column].dtype) for column in identity_columns],
        "index_dtype": str(identity.index.dtype),
        "index_name": identity.index.name,
    }
    digest = hashlib.sha256(_canonical_json(header).encode())
    digest.update(
        pd.util.hash_pandas_object(identity, index=True).to_numpy(dtype="<u8").tobytes()
    )
    return digest.hexdigest()


def _ordered_strings_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode()
    ).hexdigest()


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(_canonical_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
