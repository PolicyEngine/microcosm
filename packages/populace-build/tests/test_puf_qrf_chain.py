"""Per-target subprocess checkpoints for the primary PUF QRF chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_TARGET_ORDER,
    PRIMARY_QRF_TARGET_ORDER_SHA256,
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    load_primary_puf_qrf_predictions,
    run_primary_puf_qrf_chain,
)
from populace.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    clone_us_frame_for_puf_support,
    finalize_us_puf_tax_detail_predictions,
    prepare_us_puf_tax_detail_chain_inputs,
)
from populace.fit import QRF
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

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
    assert len(PRIMARY_QRF_TARGET_ORDER) == 64
    digest = hashlib.sha256(
        json.dumps(list(PRIMARY_QRF_TARGET_ORDER), separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == PRIMARY_QRF_TARGET_ORDER_SHA256


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


def test_primary_qrf_rejects_pre_carve_schema_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # populace#515: the checkpointed donor frame carries the E19200 ->
    # mortgage-only carve as of schema v2, and loading validates nothing
    # about donor construction identity -- so a pre-carve (v1) root manifest
    # OR a v1 target checkpoint riding under a valid v2 root must both be
    # rejected, never fitted or drawn from.
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
    assert original_manifest["schema_version"] == 2
    stale_manifest = dict(original_manifest)
    stale_manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(stale_manifest))
    with pytest.raises(ValueError, match="schema version"):
        load_primary_puf_qrf_predictions(checkpoint_dir)
    with pytest.raises(ValueError, match="schema version"):
        run_primary_puf_qrf_chain(checkpoint_dir)
    manifest_path.write_text(json.dumps(original_manifest))

    target_path = sorted((checkpoint_dir / "targets").glob("*.h5"))[0]
    with h5py.File(target_path, mode="r+") as h5:
        metadata = json.loads(bytes(h5["metadata_json"][...]).decode())
        assert metadata["schema_version"] == 2
        metadata["schema_version"] = 1
        del h5["metadata_json"]
        h5.create_dataset(
            "metadata_json",
            data=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
            track_times=False,
        )
    with pytest.raises(ValueError, match="invalid schema_version"):
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
