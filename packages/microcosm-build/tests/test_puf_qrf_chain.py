"""Per-target subprocess checkpoints for the primary PUF QRF chain."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.puf_qrf_chain as puf_qrf_chain_module
from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.serialization_dtypes import (
    CANONICAL_STRING_DTYPE,
    canonicalize_frame_string_dtypes,
)
from microcosm.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
    PRIMARY_QRF_TARGET_ORDER,
    PRIMARY_QRF_TARGET_ORDER_SHA256,
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    load_primary_puf_qrf_predictions,
    run_primary_puf_qrf_chain,
    run_primary_puf_qrf_target,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_LEGACY_ZERO_FILL,
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    clone_us_frame_for_puf_support,
    finalize_us_puf_tax_detail_predictions,
    prepare_us_puf_tax_detail_chain_inputs,
)
from microcosm.fit import QRF
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_PREDICTORS = (
    "puf_predictor_filing_status_code",
    "puf_predictor_tax_unit_person_count",
)
_PERSON_OUTPUTS = (
    "taxable_interest_income",
    "qualified_tuition_expenses",
)
_TAX_UNIT_OUTPUTS = ("domestic_production_ald",)


def _expanded_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_household_id": np.asarray([1, 1, 2], dtype=np.int64),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype=np.int64),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype=np.int64),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype=np.int64),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype=np.int64),
        },
        index=pd.Index([101, 103, 107], name="source_person_row"),
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype=np.int64),
                "state_fips": np.asarray([6, 36], dtype=np.int64),
            },
            index=pd.Index([201, 203], name="source_household_row"),
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype=np.int64),
                "filing_status_input": ["JOINT", "SINGLE"],
            },
            index=pd.Index([301, 307], name="source_tax_unit_row"),
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    return clone_us_frame_for_puf_support(
        Frame(
            tables,
            US_SCHEMA,
            {"household": Weights(np.asarray([100.0, 300.0]), WeightKind.DESIGN)},
            pd.Series(
                ["asec_2024", "asec_2024", "asec_2023"],
                index=person.index,
                name="stratum",
            ),
        )
    )


def _donor() -> pd.DataFrame:
    rng = np.random.default_rng(1207)
    rows = 40
    filing_status = rng.choice([1.0, 2.0, 4.0], size=rows)
    person_count = rng.choice([1.0, 2.0, 3.0], size=rows)
    interest = np.where(
        rng.random(rows) < 0.45,
        0.0,
        rng.lognormal(mean=5.0, sigma=0.8, size=rows),
    )
    tuition = np.where(
        rng.random(rows) < 0.7,
        0.0,
        0.25 * interest + rng.lognormal(mean=4.0, sigma=0.5, size=rows),
    )
    domestic = np.where(
        rng.random(rows) < 0.8,
        0.0,
        0.1 * tuition + rng.lognormal(mean=3.0, sigma=0.4, size=rows),
    )
    return pd.DataFrame(
        {
            "filing_status_code": filing_status,
            "tax_unit_person_count": person_count,
            "taxable_interest_income": interest,
            "qualified_tuition_expenses": tuition,
            "domestic_production_ald": domestic,
            "weight": rng.uniform(1.0, 10.0, size=rows),
        }
    )


def _assert_frames_bit_exact(actual: Frame, expected: Frame) -> None:
    assert actual.schema == expected.schema
    assert actual.entities == expected.entities
    for entity in expected.entities:
        actual_table = actual.table(entity)
        expected_table = expected.table(entity)
        pd.testing.assert_frame_equal(
            actual_table,
            expected_table,
            check_dtype=True,
            check_exact=True,
        )
        for column in expected_table.select_dtypes(include=["floating"]):
            np.testing.assert_array_equal(
                actual_table[column].to_numpy().view(np.uint64),
                expected_table[column].to_numpy().view(np.uint64),
            )
    for entity in expected.weighted_entities:
        assert actual.weights_for(entity).kind is expected.weights_for(entity).kind
        np.testing.assert_array_equal(
            actual.weights_for(entity).values,
            expected.weights_for(entity).values,
        )
    pd.testing.assert_series_equal(actual.strata, expected.strata, check_exact=True)
    assert actual.mass_log == expected.mass_log


def test_primary_qrf_production_target_order_is_locked() -> None:
    expected = (
        *PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    )
    assert PRIMARY_QRF_TARGET_ORDER == expected
    assert len(PRIMARY_QRF_TARGET_ORDER) == 65
    assert "investment_interest_expense" in PRIMARY_QRF_TARGET_ORDER
    digest = hashlib.sha256(
        json.dumps(list(PRIMARY_QRF_TARGET_ORDER), separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == PRIMARY_QRF_TARGET_ORDER_SHA256


def test_primary_qrf_manifest_fsyncs_file_then_parent_directory_after_rename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(puf_qrf_chain_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(puf_qrf_chain_module.os, "replace", tracked_replace)
    manifest_path = tmp_path / "manifest.json"

    puf_qrf_chain_module._atomic_write_json(manifest_path, {"fixture": True})

    assert events == [
        ("fsync", "file"),
        ("replace", manifest_path.name),
        ("fsync", "directory"),
    ]


def test_primary_qrf_frame_bank_writes_canonical_string_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _expanded_frame()
    support_channel = "tax_unit_support_channel"
    frame.table("tax_unit")[support_channel] = frame.table("tax_unit")[
        support_channel
    ].astype(object)
    assert pd.api.types.is_object_dtype(frame.table("tax_unit")[support_channel].dtype)

    real_prepare = puf_qrf_chain_module.prepare_us_puf_tax_detail_chain_inputs

    def prepare_with_object_donor_label(*args, **kwargs):
        inputs = real_prepare(*args, **kwargs)
        donor = inputs.donor_frame.table("tax_unit")
        donor["bank_fixture_label"] = pd.Series(
            np.asarray(["PUF"] * len(donor), dtype=object),
            index=donor.index,
        )
        return inputs

    monkeypatch.setattr(
        puf_qrf_chain_module,
        "prepare_us_puf_tax_detail_chain_inputs",
        prepare_with_object_donor_label,
    )
    checkpoint_dir = tmp_path / "primary_qrf"
    manifest = initialize_primary_puf_qrf_chain(
        frame,
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
    )

    donor = load_frame_checkpoint(
        checkpoint_dir / puf_qrf_chain_module.PRIMARY_QRF_DONOR_FILENAME
    ).frame
    recipient = load_frame_checkpoint(
        checkpoint_dir / puf_qrf_chain_module.PRIMARY_QRF_RECIPIENT_FILENAME
    ).frame
    assert donor.table("tax_unit")["bank_fixture_label"].dtype == (
        CANONICAL_STRING_DTYPE
    )
    assert recipient.table("tax_unit")[support_channel].dtype == (
        CANONICAL_STRING_DTYPE
    )
    identity_columns = tuple(manifest["recipient_identity_columns"])
    assert (
        puf_qrf_chain_module._recipient_identity_sha256(
            recipient.table("tax_unit"), identity_columns
        )
        == manifest["recipient_identity_sha256"]
    )


def test_primary_qrf_legacy_object_bank_loads_without_rewrite_or_identity_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame = _expanded_frame()
    support_channel = "tax_unit_support_channel"
    frame.table("tax_unit")[support_channel] = frame.table("tax_unit")[
        support_channel
    ].astype(object)
    checkpoint_dir = tmp_path / "legacy_primary_qrf"
    with monkeypatch.context() as legacy_policy:
        legacy_policy.setattr(
            puf_qrf_chain_module,
            "canonicalize_frame_string_dtypes",
            lambda frame, **_kwargs: frame,
        )
        manifest = initialize_primary_puf_qrf_chain(
            frame,
            _donor(),
            checkpoint_dir,
            predictors=_PREDICTORS,
            person_outputs=_PERSON_OUTPUTS,
            tax_unit_outputs=_TAX_UNIT_OUTPUTS,
            n_estimators=2,
            seed=3,
        )

    donor_path = checkpoint_dir / puf_qrf_chain_module.PRIMARY_QRF_DONOR_FILENAME
    recipient_path = (
        checkpoint_dir / puf_qrf_chain_module.PRIMARY_QRF_RECIPIENT_FILENAME
    )
    bank_digests = {
        donor_path: puf_qrf_chain_module._file_sha256(donor_path),
        recipient_path: puf_qrf_chain_module._file_sha256(recipient_path),
    }
    raw_recipient = load_frame_checkpoint(recipient_path).frame.table("tax_unit")
    assert pd.api.types.is_object_dtype(raw_recipient[support_channel].dtype)
    identity_columns = tuple(manifest["recipient_identity_columns"])
    assert (
        puf_qrf_chain_module._recipient_identity_sha256(raw_recipient, identity_columns)
        == manifest["recipient_identity_sha256"]
    )
    loaded_recipient = puf_qrf_chain_module._load_bound_frame(
        checkpoint_dir,
        manifest,
        filename_key="recipient_filename",
        digest_key="recipient_checkpoint_sha256",
        role="recipient",
    )
    assert (
        loaded_recipient.table("tax_unit")[support_channel].dtype
        == CANONICAL_STRING_DTYPE
    )

    live = canonicalize_frame_string_dtypes(
        frame,
        boundary="primary QRF legacy-bank test live frame",
    )
    assert live.table("tax_unit")[support_channel].dtype == CANONICAL_STRING_DTYPE
    for target_index in range(len((*_PERSON_OUTPUTS, *_TAX_UNIT_OUTPUTS))):
        run_primary_puf_qrf_target(checkpoint_dir, target_index)
    _finalized, weight_kind = finalize_primary_puf_qrf_chain(live, checkpoint_dir)

    assert weight_kind == "design"
    assert {
        path: puf_qrf_chain_module._file_sha256(path) for path in bank_digests
    } == bank_digests


def test_primary_qrf_target_fsyncs_file_then_parent_directory_after_rename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    checkpoint_dir = tmp_path / "primary_qrf"
    initialize_primary_puf_qrf_chain(
        _expanded_frame(),
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
    )
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(puf_qrf_chain_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(puf_qrf_chain_module.os, "replace", tracked_replace)
    monkeypatch.setattr(
        puf_qrf_chain_module,
        "profile_stage",
        lambda *_args, **_kwargs: nullcontext(),
    )

    target_path = run_primary_puf_qrf_target(checkpoint_dir, 0)

    assert target_path.is_file()
    assert events == [
        ("fsync", "file"),
        ("replace", target_path.name),
        ("fsync", "directory"),
    ]


def test_target_subprocess_chain_matches_monolith_raw_bits_and_final_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame = _expanded_frame()
    donor = _donor()
    inputs = prepare_us_puf_tax_detail_chain_inputs(
        frame,
        donor,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
    )
    monolith_raw = (
        QRF(n_estimators=4, seed=17)
        .fit(
            inputs.donor_frame,
            list(_PREDICTORS),
            list(inputs.target_order),
            weights="design",
        )
        .predict(inputs.recipient_features)
    )

    checkpoint_dir = tmp_path / "primary_qrf"
    initialize_primary_puf_qrf_chain(
        frame,
        donor,
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=4,
        seed=17,
    )
    run_primary_puf_qrf_chain(
        checkpoint_dir,
        environment={
            "POPULACE_FIT_N_JOBS": "1",
            "POPULACE_FIT_PREDICT_WORKERS": "1",
        },
    )
    staged_raw = load_primary_puf_qrf_predictions(checkpoint_dir)

    assert list(staged_raw.columns) == list(monolith_raw.columns)
    assert staged_raw.index.equals(monolith_raw.index)
    for target in monolith_raw:
        np.testing.assert_array_equal(
            staged_raw[target].to_numpy().view(np.uint64),
            monolith_raw[target].to_numpy().view(np.uint64),
        )

    expected = finalize_us_puf_tax_detail_predictions(
        frame,
        inputs.donor,
        monolith_raw.copy(),
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
    )
    actual, weight_kind = finalize_primary_puf_qrf_chain(frame, checkpoint_dir)
    assert weight_kind == "design"
    _assert_frames_bit_exact(actual, expected)

    tax_unit = frame.table("tax_unit")
    puf_rows = tax_unit["tax_unit_support_channel"] == "puf_tax_detail"
    first_puf_row = tax_unit.index[puf_rows][0]
    original_source_id = tax_unit.loc[first_puf_row, "tax_unit_source_id"]
    tax_unit.loc[first_puf_row, "tax_unit_source_id"] = original_source_id + 1
    with pytest.raises(ValueError, match="changed PUF recipient identity"):
        finalize_primary_puf_qrf_chain(frame, checkpoint_dir)
    tax_unit.loc[first_puf_row, "tax_unit_source_id"] = original_source_id

    # Finalization must not rewrite raw target checkpoints. A second supervisor
    # pass validates and skips the complete contiguous prefix.
    raw_after_finalization = load_primary_puf_qrf_predictions(checkpoint_dir)
    for target in staged_raw:
        np.testing.assert_array_equal(
            raw_after_finalization[target].to_numpy().view(np.uint64),
            staged_raw[target].to_numpy().view(np.uint64),
        )
    run_primary_puf_qrf_chain(checkpoint_dir)


def test_resumed_earnings_chain_keeps_children_out_of_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame = _expanded_frame()
    person = frame.table("person")
    source_age = {1: 12.0, 2: 40.0, 3: 10.0}
    person["age"] = person["person_source_id"].map(source_age).astype("float64")
    donor = _donor()
    donor["employment_income_before_lsr"] = np.linspace(
        1_000.0,
        40_000.0,
        len(donor),
    )
    donor["self_employment_income_before_lsr"] = np.linspace(
        100.0,
        4_000.0,
        len(donor),
    )
    checkpoint_dir = tmp_path / "primary_qrf_earnings_resume"
    person_outputs = (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    )
    manifest = initialize_primary_puf_qrf_chain(
        frame,
        donor,
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=person_outputs,
        tax_unit_outputs=(),
        n_estimators=2,
        seed=19,
        require_complete_recipient_predictors=True,
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    allocation = manifest["recipient_predictor_universe"]["person_output_allocation"]
    assert allocation["person_outputs"] == sorted(person_outputs)
    assert allocation["out_of_universe_person_rows"] == 2
    assert allocation["empty_eligible_tax_unit_rows"] == 1
    assert allocation["first_person_fallback_out_of_universe_rows"] == 0

    completed_target = run_primary_puf_qrf_target(checkpoint_dir, 0)
    completed_bytes = completed_target.read_bytes()
    pending_target = puf_qrf_chain_module._target_path(checkpoint_dir, manifest, 1)
    assert not pending_target.exists()

    # The supervisor preserves the completed prefix and resumes the missing target.
    run_primary_puf_qrf_chain(checkpoint_dir)
    assert completed_target.read_bytes() == completed_bytes
    assert pending_target.is_file()
    resumed, weight_kind = finalize_primary_puf_qrf_chain(frame, checkpoint_dir)

    assert weight_kind == "design"
    resumed_person = resumed.table("person")
    detail = resumed_person["person_support_channel"].eq("puf_tax_detail")
    children = detail & resumed_person["age"].lt(15)
    adults = detail & resumed_person["age"].ge(15)
    assert int(children.sum()) == 2
    assert resumed_person.loc[children, list(person_outputs)].eq(0.0).all().all()
    assert resumed_person.loc[adults, list(person_outputs)].gt(0.0).all().all()
    all_child_totals = (
        resumed_person.loc[detail]
        .groupby("person_tax_unit_id", sort=False)[list(person_outputs)]
        .sum()
    )
    assert all_child_totals.loc[100020].eq(0.0).all()


def test_primary_qrf_chain_manifest_binds_stacked_doctrines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    checkpoint_dir = tmp_path / "primary_qrf"
    manifest = initialize_primary_puf_qrf_chain(
        _expanded_frame(),
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
        require_complete_recipient_predictors=True,
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )

    expected_controls = {
        "require_complete_recipient_predictors": True,
        "absent_cells": PUF_ABSENT_CELLS_PRESERVE_NULLS,
        "recipient_predictor_universe": manifest["recipient_predictor_universe"],
    }
    assert len(expected_controls["recipient_predictor_universe"]["sha256"]) == 64
    assert {name: manifest[name] for name in expected_controls} == expected_controls
    for filename in (
        puf_qrf_chain_module.PRIMARY_QRF_DONOR_FILENAME,
        puf_qrf_chain_module.PRIMARY_QRF_RECIPIENT_FILENAME,
    ):
        checkpoint = load_frame_checkpoint(checkpoint_dir / filename)
        assert {
            name: checkpoint.metadata[name] for name in expected_controls
        } == expected_controls

    target_path = run_primary_puf_qrf_target(checkpoint_dir, 0)
    with h5py.File(target_path, mode="r") as h5:
        target_receipt = json.loads(bytes(h5["metadata_json"][...]).decode())
    assert {
        name: target_receipt[name] for name in expected_controls
    } == expected_controls

    manifest_path = checkpoint_dir / "manifest.json"
    tampered_manifest = dict(manifest)
    tampered_manifest["absent_cells"] = PUF_ABSENT_CELLS_LEGACY_ZERO_FILL
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="doctrine controls"):
        run_primary_puf_qrf_target(checkpoint_dir, 1)

    tampered_manifest = dict(manifest)
    tampered_universe = dict(manifest["recipient_predictor_universe"])
    tampered_universe["recipient_tax_unit_rows"] += 1
    tampered_manifest["recipient_predictor_universe"] = tampered_universe
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="universe digest is invalid"):
        run_primary_puf_qrf_target(checkpoint_dir, 1)


def test_primary_qrf_chain_stacked_finalization_preserves_unowned_nulls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame = _expanded_frame()
    frame.table("person")["taxable_interest_income"] = np.nan
    frame.table("tax_unit")["domestic_production_ald"] = np.nan
    checkpoint_dir = tmp_path / "primary_qrf"
    initialize_primary_puf_qrf_chain(
        frame,
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
        require_complete_recipient_predictors=True,
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    for target_index in range(len((*_PERSON_OUTPUTS, *_TAX_UNIT_OUTPUTS))):
        run_primary_puf_qrf_target(checkpoint_dir, target_index)

    finalized, weight_kind = finalize_primary_puf_qrf_chain(frame, checkpoint_dir)

    assert weight_kind == "design"
    person = finalized.table("person")
    person_clone = person["person_support_clone_index"]
    assert person.loc[person_clone.eq(0), "taxable_interest_income"].isna().all()
    assert person.loc[person_clone.eq(1), "taxable_interest_income"].notna().all()
    tax_unit = finalized.table("tax_unit")
    tax_unit_clone = tax_unit["tax_unit_support_clone_index"]
    assert tax_unit.loc[tax_unit_clone.eq(0), "domestic_production_ald"].isna().all()
    assert tax_unit.loc[tax_unit_clone.eq(1), "domestic_production_ald"].notna().all()


def test_primary_qrf_chain_legacy_doctrine_defaults_remain_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame = _expanded_frame()
    frame.table("person")["taxable_interest_income"] = np.nan
    checkpoint_dir = tmp_path / "legacy_primary_qrf"
    manifest = initialize_primary_puf_qrf_chain(
        frame,
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
    )
    assert manifest["schema_version"] == 6
    assert "require_complete_recipient_predictors" not in manifest
    assert "absent_cells" not in manifest
    assert "recipient_predictor_universe" not in manifest
    for filename in (
        puf_qrf_chain_module.PRIMARY_QRF_DONOR_FILENAME,
        puf_qrf_chain_module.PRIMARY_QRF_RECIPIENT_FILENAME,
    ):
        metadata = load_frame_checkpoint(checkpoint_dir / filename).metadata
        assert "require_complete_recipient_predictors" not in metadata
        assert "absent_cells" not in metadata
        assert "recipient_predictor_universe" not in metadata
    for target_index in range(len((*_PERSON_OUTPUTS, *_TAX_UNIT_OUTPUTS))):
        run_primary_puf_qrf_target(checkpoint_dir, target_index)

    finalized, _weight_kind = finalize_primary_puf_qrf_chain(frame, checkpoint_dir)

    person = finalized.table("person")
    native = person["person_support_clone_index"].eq(0)
    assert person.loc[native, "taxable_interest_income"].eq(0.0).all()


def test_primary_qrf_chain_rejects_incomplete_stacked_predictors(
    tmp_path: Path,
) -> None:
    frame = _expanded_frame()
    tax_unit = frame.table("tax_unit")
    puf_row = tax_unit.index[tax_unit["tax_unit_support_clone_index"].eq(1)][0]
    tax_unit.loc[puf_row, "filing_status_input"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing values before coercion.*puf_predictor_filing_status_code",
    ):
        initialize_primary_puf_qrf_chain(
            frame,
            _donor(),
            tmp_path / "primary_qrf",
            predictors=_PREDICTORS,
            person_outputs=_PERSON_OUTPUTS,
            tax_unit_outputs=_TAX_UNIT_OUTPUTS,
            n_estimators=2,
            seed=3,
            require_complete_recipient_predictors=True,
            absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
        )
    assert not (tmp_path / "primary_qrf").exists()


def test_primary_qrf_finalization_rejects_changed_recipient_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame = _expanded_frame()
    checkpoint_dir = tmp_path / "primary_qrf"
    initialize_primary_puf_qrf_chain(
        frame,
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
        require_complete_recipient_predictors=True,
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )
    for target_index in range(len((*_PERSON_OUTPUTS, *_TAX_UNIT_OUTPUTS))):
        run_primary_puf_qrf_target(checkpoint_dir, target_index)
    tax_unit = frame.table("tax_unit")
    detail = tax_unit["tax_unit_support_clone_index"].eq(1)
    row = tax_unit.index[detail][0]
    tax_unit.loc[row, "filing_status_input"] = "SINGLE"

    with pytest.raises(
        ValueError,
        match="changed the PUF recipient predictor source universe or feature values",
    ):
        finalize_primary_puf_qrf_chain(frame, checkpoint_dir)


@pytest.mark.parametrize("stale_version", (1, 2, 3, 4, 5))
def test_primary_qrf_rejects_every_stale_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_version: int,
) -> None:
    # Roots and target checkpoints must reject every schema predating the v6
    # recipient-universe authority. In particular, the v5 case models the old
    # strict two-control payload that omitted recipient_predictor_universe; its
    # stale schema is rejected before that payload can be interpreted.
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    checkpoint_dir = tmp_path / "primary_qrf"
    initialize_primary_puf_qrf_chain(
        _expanded_frame(),
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
    )
    run_primary_puf_qrf_chain(
        checkpoint_dir,
        environment={
            "POPULACE_FIT_N_JOBS": "1",
            "POPULACE_FIT_PREDICT_WORKERS": "1",
        },
    )

    manifest_path = checkpoint_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text())
    assert original_manifest["schema_version"] == PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION
    stale_manifest = dict(original_manifest)
    stale_manifest["schema_version"] = stale_version
    if stale_version == 5:
        stale_manifest["require_complete_recipient_predictors"] = True
        stale_manifest["absent_cells"] = PUF_ABSENT_CELLS_PRESERVE_NULLS
    manifest_path.write_text(json.dumps(stale_manifest))
    with pytest.raises(
        ValueError,
        match=rf"schema version: expected 6, got {stale_version}",
    ):
        load_primary_puf_qrf_predictions(checkpoint_dir)
    with pytest.raises(
        ValueError,
        match=rf"schema version: expected 6, got {stale_version}",
    ):
        run_primary_puf_qrf_chain(checkpoint_dir)
    manifest_path.write_text(json.dumps(original_manifest))

    target_path = sorted((checkpoint_dir / "targets").glob("*.h5"))[0]
    with h5py.File(target_path, mode="r") as h5:
        pristine_metadata = json.loads(bytes(h5["metadata_json"][...]).decode())
    assert pristine_metadata["schema_version"] == PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION
    metadata = dict(pristine_metadata)
    metadata["schema_version"] = stale_version
    if stale_version == 5:
        metadata["require_complete_recipient_predictors"] = True
        metadata["absent_cells"] = PUF_ABSENT_CELLS_PRESERVE_NULLS
    with h5py.File(target_path, mode="r+") as h5:
        del h5["metadata_json"]
        h5.create_dataset(
            "metadata_json",
            data=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
            track_times=False,
        )
    with pytest.raises(
        ValueError,
        match=rf"invalid schema_version: expected 6, got {stale_version}",
    ):
        load_primary_puf_qrf_predictions(checkpoint_dir)


def test_primary_qrf_resume_rejects_a_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    checkpoint_dir = tmp_path / "primary_qrf"
    initialize_primary_puf_qrf_chain(
        _expanded_frame(),
        _donor(),
        checkpoint_dir,
        predictors=_PREDICTORS,
        person_outputs=_PERSON_OUTPUTS,
        tax_unit_outputs=_TAX_UNIT_OUTPUTS,
        n_estimators=2,
        seed=3,
    )
    manifest_path = checkpoint_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text())
    invalid_manifest = dict(original_manifest)
    invalid_manifest["person_outputs"] = list(reversed(_PERSON_OUTPUTS))
    manifest_path.write_text(json.dumps(invalid_manifest))
    with pytest.raises(ValueError, match="output partition"):
        load_primary_puf_qrf_predictions(checkpoint_dir)
    manifest_path.write_text(json.dumps(original_manifest))
    run_primary_puf_qrf_chain(
        checkpoint_dir,
        environment={
            "POPULACE_FIT_N_JOBS": "1",
            "POPULACE_FIT_PREDICT_WORKERS": "1",
        },
    )
    target_paths = sorted((checkpoint_dir / "targets").glob("*.h5"))
    assert len(target_paths) == 3
    with h5py.File(target_paths[0], mode="r+") as h5:
        original_bits = np.asarray(h5["raw_draw_bits"])[0]
        h5["raw_draw_bits"][0] = original_bits ^ np.uint64(1)
    with pytest.raises(ValueError, match="raw draw digest"):
        load_primary_puf_qrf_predictions(checkpoint_dir)
    with h5py.File(target_paths[0], mode="r+") as h5:
        h5["raw_draw_bits"][0] = original_bits
    target_paths[1].unlink()

    with pytest.raises(ValueError, match="non-contiguous prefix"):
        run_primary_puf_qrf_chain(checkpoint_dir)
