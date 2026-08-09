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

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.serialization_dtypes import (
    canonicalize_frame_string_dtypes,
    canonicalize_table_string_dtypes,
)
from microcosm.build.stage_profile import profile_stage
from microcosm.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PufTaxDetailChainInputs,
    finalize_us_puf_tax_detail_predictions,
    prepare_us_puf_tax_detail_chain_inputs,
    puf_recipient_predictor_universe_receipt,
)
from microcosm.build.us_runtime.support_provenance import (
    puf_tax_detail_clone_mask,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.fit import QRF, QRFChainState
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

# v2 (microcosm#515): the checkpointed donor frame gained the interim national
# E19200 -> mortgage-only concept carve. Loading validates only
# schema/digest/kind/role -- not donor construction identity -- so a v1
# checkpoint initialized pre-carve would keep fitting and drawing the uncarved
# levels under carved code.
# v3 (microcosm#516): donor construction now whole-row-screens grouped raw
# mortgage-interest outliers before that carve. A v2 post-carve, pre-screen
# checkpoint would otherwise still fit and draw the corrupt rows under screened
# code, so it too must be rejected and rebuilt.
# v4 (microcosm#515 completion): donor construction replaces the national carve
# with published per-AGI-band shares and adds investment_interest_expense as a
# learned person input. The production target-order digest also changes, but a
# schema bump is still required for standalone custom-order checkpoints whose
# manifests do not fingerprint donor-construction semantics.
# v5 (microcosm#567): the grouped-raw mortgage screen is field-local instead of
# dropping whole donor rows. A v4 checkpoint would silently retain the
# whole-row quarantine semantics and its missing capital-gains support.
# microcosm#578 does not bump v5: object-backed and canonical StringDtype
# support-channel columns are two physical encodings of the same identity-only
# values, never QRF predictors or targets. Loads authenticate the immutable file
# SHA, metadata, and legacy dtype-sensitive recipient identity before applying
# the canonical in-memory string policy, so existing bank bytes remain valid.
# The microcosm#578 stacked-spine doctrine controls likewise do not bump v5:
# legacy manifests omit both fields and retain their exact bytes and behavior.
# A non-legacy initialization writes both controls into each immutable bank,
# the root manifest, and every target receipt so deletion or mutation cannot
# silently downgrade a stacked chain to the legacy policy.
# Strict banks also bind the exact recipient source-universe/feature receipt;
# its optional v5 field invalidates pre-declaration strict banks without
# changing byte-identical legacy v5 manifests.
# v6 makes that recipient-universe authority a versioned chain semantic. Every
# v1--v5 root or target must rebuild rather than sharing a schema label with a
# chain whose root, banks, targets, and finalization bind the added receipt.
PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION = 6
PRIMARY_QRF_MANIFEST_FILENAME = "manifest.json"
PRIMARY_QRF_DONOR_FILENAME = "donor.frame.h5"
PRIMARY_QRF_RECIPIENT_FILENAME = "recipient.frame.h5"
PRIMARY_QRF_TARGETS_DIRNAME = "targets"
PRIMARY_QRF_TARGET_ORDER = (
    *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
PRIMARY_QRF_TARGET_ORDER_SHA256 = (
    "795519d161e6b8425fc3b64de7eb435d52d25e7c8250b5861f3bb21ab48266a3"
)

_ARTIFACT_KIND = "populace_primary_puf_qrf_chain"
_RAW_TARGET_ARTIFACT_KIND = "populace_primary_puf_qrf_raw_target"
_METADATA_DATASET = "metadata_json"
_RAW_DRAW_BITS_DATASET = "raw_draw_bits"
_REQUIRE_COMPLETE_RECIPIENT_PREDICTORS = "require_complete_recipient_predictors"
_ABSENT_CELLS = "absent_cells"
_RECIPIENT_PREDICTOR_UNIVERSE = "recipient_predictor_universe"
_ABSENT_CELLS_POLICIES = (
    PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
)

__all__ = [
    "PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION",
    "PRIMARY_QRF_TARGET_ORDER",
    "PRIMARY_QRF_TARGET_ORDER_SHA256",
    "finalize_primary_puf_qrf_chain",
    "initialize_primary_puf_qrf_chain",
    "load_primary_puf_qrf_predictions",
    "primary_puf_qrf_recipient_predictor_universe_receipt",
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
    require_complete_recipient_predictors: bool = False,
    absent_cells: str = PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
) -> dict[str, object]:
    """Write immutable donor/recipient inputs and the initial RNG manifest.

    The defaults preserve the historical two-spine chain exactly: recipient
    predictor absence is zero-filled, finalization globally zero-fills absent
    outputs, and the three doctrine fields are omitted so historical legacy-bank
    payload bytes do not change. Selecting either non-legacy control declares both settings
    plus the recipient source-universe receipt in every immutable bank, the
    root manifest, and per-target receipts.
    """

    require_complete_recipient_predictors, absent_cells = (
        _validate_chain_doctrine_controls(
            require_complete_recipient_predictors,
            absent_cells,
            source="Primary QRF initialization",
        )
    )
    root = Path(checkpoint_dir)
    manifest_path = root / PRIMARY_QRF_MANIFEST_FILENAME
    if manifest_path.exists():
        raise FileExistsError(f"Primary QRF manifest already exists: {manifest_path}")

    preparation_kwargs: dict[str, object] = {}
    if require_complete_recipient_predictors:
        preparation_kwargs[_REQUIRE_COMPLETE_RECIPIENT_PREDICTORS] = True
    inputs = prepare_us_puf_tax_detail_chain_inputs(
        frame,
        donor_tax_units,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
        **preparation_kwargs,
    )
    doctrine_receipt = _declared_chain_doctrine_receipt(
        require_complete_recipient_predictors,
        absent_cells,
        inputs.recipient_predictor_universe,
    )
    target_order = inputs.target_order
    if (
        tuple(person_outputs) == PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
        and tuple(tax_unit_outputs) == PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
        and _ordered_strings_sha256(target_order) != PRIMARY_QRF_TARGET_ORDER_SHA256
    ):
        raise AssertionError("The production primary QRF target order changed.")

    # Preflight is intentionally complete before any checkpoint path exists:
    # a failed source-universe/completeness contract cannot leave a poisoned,
    # nonempty root that looks resumable to the stacked supervisor.
    root.mkdir(parents=True, exist_ok=True)
    (root / PRIMARY_QRF_TARGETS_DIRNAME).mkdir(exist_ok=True)

    donor_frame = canonicalize_frame_string_dtypes(
        inputs.donor_frame,
        boundary="primary PUF QRF donor bank write",
        in_place=True,
    )
    donor_path = root / PRIMARY_QRF_DONOR_FILENAME
    write_frame_checkpoint(
        donor_path,
        donor_frame,
        metadata={
            "artifact_kind": _ARTIFACT_KIND,
            "role": "donor",
            "target_order": list(target_order),
            **doctrine_receipt,
        },
    )
    recipient_frame, identity_columns = _recipient_checkpoint_frame(frame, inputs)
    recipient_frame = canonicalize_frame_string_dtypes(
        recipient_frame,
        boundary="primary PUF QRF recipient bank write",
        in_place=True,
    )
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
            **doctrine_receipt,
        },
    )

    model = QRF(n_estimators=n_estimators, seed=seed)
    state = model.start_chain(
        donor_frame,
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
        **doctrine_receipt,
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
                "microcosm.build.us_runtime.puf_qrf_worker",
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


def primary_puf_qrf_recipient_predictor_universe_receipt(
    checkpoint_dir: str | Path,
) -> dict[str, object]:
    """Load the bank-authenticated strict recipient-universe receipt."""

    root = Path(checkpoint_dir).resolve()
    manifest = _load_manifest(root)
    # Authenticate the recipient bank and require its doctrine fields to agree
    # before exposing the manifest copy to an outer stacked receipt.
    _load_bound_frame(
        root,
        manifest,
        filename_key="recipient_filename",
        digest_key="recipient_checkpoint_sha256",
        role="recipient",
    )
    require_complete, _absent_cells, universe, declared = (
        _resolve_chain_doctrine_controls(
            manifest,
            source="Primary QRF manifest",
        )
    )
    if not declared or not require_complete or not universe:
        raise ValueError(
            "Primary QRF checkpoint is not a strict recipient-universe bank."
        )
    return json.loads(json.dumps(universe))


def finalize_primary_puf_qrf_chain(
    frame: Frame,
    checkpoint_dir: str | Path,
    *,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
) -> tuple[Frame, str]:
    """Finalize all raw checkpoints onto ``frame`` and return fit weight kind."""

    root = Path(checkpoint_dir).resolve()
    manifest = _load_manifest(root)
    _assert_live_recipient_identity(frame, root, manifest)
    _assert_live_recipient_predictor_universe(frame, manifest)
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
        **_finalization_doctrine_kwargs(manifest),
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
    expected_metadata: dict[str, object] = {
        "artifact_kind": _ARTIFACT_KIND,
        "role": role,
        "target_order": list(_manifest_strings(manifest, "target_order")),
    }
    if role == "recipient":
        expected_metadata.update(
            {
                "identity_columns": list(
                    _manifest_strings(manifest, "recipient_identity_columns")
                ),
                "predictors": list(_manifest_strings(manifest, "predictors")),
            }
        )
    for key, expected in expected_metadata.items():
        if loaded.metadata.get(key) != expected:
            raise ValueError(
                f"Primary QRF {role} checkpoint has invalid {key}: expected "
                f"{expected!r}, got {loaded.metadata.get(key)!r}."
            )
    _assert_bound_doctrine_controls(
        manifest,
        loaded.metadata,
        role=f"{role} checkpoint",
    )
    if role == "recipient":
        identity_columns = _manifest_strings(manifest, "recipient_identity_columns")
        recipient_table = loaded.frame.table("tax_unit")
        expected_rows = _manifest_integer(manifest, "recipient_rows")
        expected_identity = manifest.get("recipient_identity_sha256")
        if (
            len(recipient_table) != expected_rows
            or not isinstance(expected_identity, str)
            or _recipient_identity_sha256(recipient_table, identity_columns)
            != expected_identity
        ):
            raise ValueError("Primary QRF recipient identity or row order changed.")
    # Preserve the authenticated bytes and dtype-sensitive legacy manifest
    # binding above. Canonicalization is deliberately post-load and in-memory:
    # v5 object-backed banks remain valid while every active worker sees the
    # same physical string dtype as a newly initialized bank.
    return canonicalize_frame_string_dtypes(
        loaded.frame,
        boundary=f"primary PUF QRF {role} bank load",
        in_place=True,
    )


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
    doctrine_receipt = _manifest_doctrine_receipt(manifest)
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
        **doctrine_receipt,
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
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
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
        # Written since v1 but validated only since the microcosm#515 carve
        # bump: a target checkpoint must carry the loader's own schema, not
        # merely ride under a valid root manifest.
        "schema_version": PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
        "target": target_order[target_index],
        "target_index": target_index,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Primary QRF target checkpoint {path} has invalid {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}."
            )
    _assert_bound_doctrine_controls(
        manifest,
        metadata,
        role=f"target checkpoint {path}",
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
        raise ValueError(
            "Unsupported primary QRF checkpoint schema version: expected "
            f"{PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION}, got "
            f"{manifest.get('schema_version')!r}."
        )
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
    _resolve_chain_doctrine_controls(manifest, source="Primary QRF manifest")
    return manifest


def _assert_live_recipient_identity(
    frame: Frame,
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    identity_columns = _manifest_strings(manifest, "recipient_identity_columns")
    tax_unit = frame.table("tax_unit")
    clone_index = support_clone_index_column("tax_unit")
    if clone_index not in tax_unit:
        raise ValueError("Live finalization frame lacks PUF clone identity.")
    missing_identity = [column for column in identity_columns if column not in tax_unit]
    if missing_identity:
        raise ValueError(
            f"Live finalization frame lacks identity columns: {missing_identity}."
        )
    live = tax_unit.loc[
        puf_tax_detail_clone_mask(tax_unit, entity="tax_unit"),
        list(identity_columns),
    ]
    expected_rows = _manifest_integer(manifest, "recipient_rows")
    if len(live) != expected_rows:
        raise ValueError(
            "Live finalization frame has a different PUF recipient surface."
        )
    # The manifest authenticates the raw stored dtype in _load_bound_frame.
    # Compare the live surface to that authenticated bank under the canonical
    # logical string policy so pre-policy object-backed banks and new canonical
    # banks share one finalization identity without a schema/version bump.
    bank = _load_bound_frame(
        root,
        manifest,
        filename_key="recipient_filename",
        digest_key="recipient_checkpoint_sha256",
        role="recipient",
    ).table("tax_unit")
    if _canonical_recipient_identity_sha256(
        live, identity_columns
    ) != _canonical_recipient_identity_sha256(bank, identity_columns):
        raise ValueError(
            "Live finalization frame changed PUF recipient identity or row order."
        )


def _assert_live_recipient_predictor_universe(
    frame: Frame,
    manifest: Mapping[str, object],
) -> None:
    require_complete, _absent_cells, expected, declared = (
        _resolve_chain_doctrine_controls(
            manifest,
            source="Primary QRF manifest",
        )
    )
    if not declared or not require_complete:
        return
    live = puf_recipient_predictor_universe_receipt(
        frame,
        predictors=_manifest_strings(manifest, "predictors"),
        person_outputs=_manifest_strings(manifest, "person_outputs"),
    )
    if live != expected:
        raise ValueError(
            "Live finalization frame changed the PUF recipient predictor "
            "source universe or feature values."
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


def _validate_chain_doctrine_controls(
    require_complete_recipient_predictors: object,
    absent_cells: object,
    *,
    source: str,
) -> tuple[bool, str]:
    if not isinstance(require_complete_recipient_predictors, bool):
        raise ValueError(
            f"{source} {_REQUIRE_COMPLETE_RECIPIENT_PREDICTORS!r} must be boolean."
        )
    if not isinstance(absent_cells, str) or absent_cells not in _ABSENT_CELLS_POLICIES:
        raise ValueError(
            f"{source} {_ABSENT_CELLS!r} must be one of "
            f"{list(_ABSENT_CELLS_POLICIES)}; got {absent_cells!r}."
        )
    return require_complete_recipient_predictors, absent_cells


def _declared_chain_doctrine_receipt(
    require_complete_recipient_predictors: bool,
    absent_cells: str,
    recipient_predictor_universe: Mapping[str, object],
) -> dict[str, object]:
    if (
        not require_complete_recipient_predictors
        and absent_cells == PUF_ABSENT_CELLS_LEGACY_ZERO_FILL
    ):
        return {}
    return {
        _REQUIRE_COMPLETE_RECIPIENT_PREDICTORS: (require_complete_recipient_predictors),
        _ABSENT_CELLS: absent_cells,
        _RECIPIENT_PREDICTOR_UNIVERSE: dict(recipient_predictor_universe),
    }


def _resolve_chain_doctrine_controls(
    payload: Mapping[str, object],
    *,
    source: str,
) -> tuple[bool, str, dict[str, object], bool]:
    has_require_complete = _REQUIRE_COMPLETE_RECIPIENT_PREDICTORS in payload
    has_absent_cells = _ABSENT_CELLS in payload
    has_universe = _RECIPIENT_PREDICTOR_UNIVERSE in payload
    if len({has_require_complete, has_absent_cells, has_universe}) != 1:
        raise ValueError(
            f"{source} doctrine controls must declare all of "
            f"{_REQUIRE_COMPLETE_RECIPIENT_PREDICTORS!r} and "
            f"{_ABSENT_CELLS!r} and {_RECIPIENT_PREDICTOR_UNIVERSE!r}, "
            "or none for a legacy-mode bank."
        )
    if not has_require_complete:
        return False, PUF_ABSENT_CELLS_LEGACY_ZERO_FILL, {}, False
    require_complete, absent_cells = _validate_chain_doctrine_controls(
        payload[_REQUIRE_COMPLETE_RECIPIENT_PREDICTORS],
        payload[_ABSENT_CELLS],
        source=source,
    )
    universe = payload[_RECIPIENT_PREDICTOR_UNIVERSE]
    if not isinstance(universe, Mapping):
        raise ValueError(
            f"{source} {_RECIPIENT_PREDICTOR_UNIVERSE!r} must be an object."
        )
    normalized = json.loads(json.dumps(universe))
    if require_complete:
        digest = normalized.get("sha256")
        body = dict(normalized)
        body.pop("sha256", None)
        if not isinstance(digest, str) or digest != _mapping_sha256(body):
            raise ValueError(
                f"{source} recipient predictor universe digest is invalid."
            )
    return require_complete, absent_cells, normalized, True


def _manifest_doctrine_receipt(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    require_complete, absent_cells, universe, declared = (
        _resolve_chain_doctrine_controls(
            manifest,
            source="Primary QRF manifest",
        )
    )
    if not declared:
        return {}
    return {
        _REQUIRE_COMPLETE_RECIPIENT_PREDICTORS: require_complete,
        _ABSENT_CELLS: absent_cells,
        _RECIPIENT_PREDICTOR_UNIVERSE: universe,
    }


def _assert_bound_doctrine_controls(
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    role: str,
) -> None:
    (
        manifest_require_complete,
        manifest_absent_cells,
        manifest_universe,
        manifest_declared,
    ) = _resolve_chain_doctrine_controls(
        manifest,
        source="Primary QRF manifest",
    )
    (
        receipt_require_complete,
        receipt_absent_cells,
        receipt_universe,
        receipt_declared,
    ) = _resolve_chain_doctrine_controls(
        receipt,
        source=f"Primary QRF {role}",
    )
    if (
        receipt_declared != manifest_declared
        or receipt_require_complete != manifest_require_complete
        or receipt_absent_cells != manifest_absent_cells
        or receipt_universe != manifest_universe
    ):
        raise ValueError(
            f"Primary QRF {role} doctrine controls disagree with the manifest."
        )


def _finalization_doctrine_kwargs(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    _require_complete, absent_cells, _universe, declared = (
        _resolve_chain_doctrine_controls(
            manifest,
            source="Primary QRF manifest",
        )
    )
    if not declared:
        return {}
    return {_ABSENT_CELLS: absent_cells}


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


def _canonical_recipient_identity_sha256(
    table: pd.DataFrame,
    identity_columns: Sequence[str],
) -> str:
    """Fingerprint recipient identity under the serialization string policy."""

    identity = canonicalize_table_string_dtypes(
        table.loc[:, list(identity_columns)],
        boundary="primary PUF QRF logical recipient identity",
        table_name="tax_unit",
    )
    return _recipient_identity_sha256(identity, identity_columns)


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
        _fsync_parent_directory(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fsync_parent_directory(path: Path) -> None:
    """Persist a completed atomic rename in its containing directory."""

    # O_DIRECTORY is not universal. A read-only directory descriptor is the
    # supported POSIX fallback; failures to open or fsync still propagate.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
