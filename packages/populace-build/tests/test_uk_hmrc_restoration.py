from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import FitWeightRecord
from populace.build.uk_runtime import hmrc_restoration
from populace.build.uk_runtime.hmrc_income import (
    HMRCIncomeSourceProvenance,
    HMRCIncomeTargetSet,
)
from populace.build.uk_runtime.hmrc_restoration import (
    CERTIFIED_UK_CANDIDATE_FILENAME,
    CERTIFIED_UK_CANDIDATE_REVISION,
    CERTIFIED_UK_CANDIDATE_SHA256,
    CERTIFIED_UK_CANDIDATE_SIZE_BYTES,
    UKCertifiedCandidateIdentity,
    UKHMRCIncomeStageTransform,
    restore_uk_hmrc_income_family,
    verify_certified_uk_candidate,
)
from populace.build.uk_runtime.national_build import (
    UKNationalDataset,
    _UKSourceFileFingerprint,
    load_uk_national_dataset,
    write_uk_national_dataset,
)
from populace.build.uk_runtime.spi_income import UKSPIIncomeImputationResult
from populace.build.uk_runtime.spi_support import (
    SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    UKSPISupportResult,
    support_channel_column,
)
from populace.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)


def _dataset() -> UKNationalDataset:
    person = {
        "person_id": [1],
        "person_household_id": [1],
        "person_benunit_id": [1],
        "gift_aid": [0.0],
        "charitable_investment_gifts": [0.0],
    }
    for column in (
        *SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
        SPI_HMRC_OTHER_INCOME_COLUMN,
        SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    ):
        person[column] = [0.0]
    return UKNationalDataset(
        person=pd.DataFrame(person),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [10.0],
            }
        ),
        time_period="2023",
    )


_TEST_SOURCE_FINGERPRINT = _UKSourceFileFingerprint(1, 2, 3, 4, 5)


def _dataset_from_source(
    path: Path,
    *,
    fingerprint: _UKSourceFileFingerprint = _TEST_SOURCE_FINGERPRINT,
) -> UKNationalDataset:
    """Model the loader-only provenance in focused restoration unit tests."""

    dataset = _dataset()
    object.__setattr__(dataset, "_source_h5", path.resolve())
    object.__setattr__(dataset, "_source_file_fingerprint", fingerprint)
    return dataset


def _candidate_identity(
    tmp_path,
    *,
    verified: bool = True,
) -> UKCertifiedCandidateIdentity:
    identity = UKCertifiedCandidateIdentity(
        path=(tmp_path / CERTIFIED_UK_CANDIDATE_FILENAME).resolve(),
        filename=CERTIFIED_UK_CANDIDATE_FILENAME,
        revision=CERTIFIED_UK_CANDIDATE_REVISION,
        sha256=CERTIFIED_UK_CANDIDATE_SHA256,
        size_bytes=CERTIFIED_UK_CANDIDATE_SIZE_BYTES,
    )
    if verified:
        object.__setattr__(
            identity,
            "_verification_token",
            hmrc_restoration._CERTIFIED_CANDIDATE_VERIFICATION_TOKEN,
        )
        object.__setattr__(
            identity,
            "_source_file_fingerprint",
            _TEST_SOURCE_FINGERPRINT,
        )
    return identity


def test_certified_candidate_verification_checks_size_and_sha(
    monkeypatch,
    tmp_path,
) -> None:
    candidate = tmp_path / CERTIFIED_UK_CANDIDATE_FILENAME
    contents = b"certified candidate"
    candidate.write_bytes(contents)
    monkeypatch.setattr(
        hmrc_restoration,
        "CERTIFIED_UK_CANDIDATE_SIZE_BYTES",
        len(contents),
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "CERTIFIED_UK_CANDIDATE_SHA256",
        hashlib.sha256(contents).hexdigest(),
    )

    identity = verify_certified_uk_candidate(candidate)

    assert identity.path == candidate.resolve()
    assert identity.size_bytes == len(contents)
    assert identity.sha256 == hashlib.sha256(contents).hexdigest()

    candidate.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="expected certified candidate size"):
        verify_certified_uk_candidate(candidate)


def test_restoration_wires_replace_qrf_materialization_and_calibration(
    monkeypatch,
    tmp_path,
) -> None:
    candidate = _candidate_identity(tmp_path)
    dataset = _dataset_from_source(candidate.path)
    mass_record = MassChangeRecord(
        entity="household",
        old_total=10.0,
        new_total=10.0,
        declared_factor=1.0,
        reason="reviewed test SPI allocation",
    )
    support_person = dataset.person.copy()
    support_person["gift_aid"] = 5.0
    support_person["charitable_investment_gifts"] = 2.0
    support_person[support_channel_column("person")] = "spi"
    support = UKSPISupportResult(
        person=support_person,
        benunit=dataset.benunit.copy(),
        household=dataset.household.copy(),
        id_multiplier=10,
        spi_household_ids=(1,),
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=(mass_record,),
        replaced_spi_households=1,
        spi_prior_mass_share=0.5,
    )
    donor_path = tmp_path / "put2223uk.tab"
    donor_path.write_bytes(b"donor")
    imputation = UKSPIIncomeImputationResult(
        person=support_person,
        fit_weight_records=(
            FitWeightRecord("stage1", "design"),
            FitWeightRecord("stage2", "importance"),
        ),
        donor_path=donor_path,
        donor_sha256=hashlib.sha256(b"donor").hexdigest(),
        donor_size_bytes=len(b"donor"),
        donor_rows=1,
        stage2_training_rows=1,
        spi_prediction_rows=1,
        reviewed_absent_stage2_outputs={},
    )
    source = HMRCIncomeSourceProvenance(
        local_path=(tmp_path / "hmrc.ods").resolve(),
        sha256="b" * 64,
        publication_url="https://www.gov.uk/test",
        ods_url="https://assets.publishing.service.gov.uk/test.ods",
        source_vintage="2023-24",
        source_tax_year="2023-24",
        source_tax_year_start=2023,
        build_period="2023",
        table_names=("Table_3_6", "Table_3_7"),
    )
    source_targets = HMRCIncomeTargetSet(source=source, targets=())
    calibration_frame = Frame(
        {
            "person": pd.DataFrame({"person_id": [1], "person_household_id": [1]}),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray([10.0]), WeightKind.CALIBRATED)},
        mass_log=(mass_record,),
    )
    calibration = SimpleNamespace(
        result=SimpleNamespace(
            frame=calibration_frame,
            weights=np.asarray([10.0]),
        )
    )
    calls: list[str] = []

    def fake_targets(*_args, **_kwargs):
        calls.append("targets")
        return source_targets

    def fake_replace(**_kwargs):
        calls.append("replace")
        return support

    def fake_impute(*_args, **_kwargs):
        calls.append("impute")
        return imputation

    def fake_materialize(*_args, **_kwargs):
        calls.append("materialize")
        return SimpleNamespace()

    def fake_calibrate(*_args, **_kwargs):
        calls.append("calibrate")
        return calibration

    monkeypatch.setattr(
        hmrc_restoration,
        "materialize_hmrc_spi_income_band_targets",
        fake_targets,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "replace_uk_spi_support_tables",
        fake_replace,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "impute_uk_spi_income_support",
        fake_impute,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "materialize_uk_hmrc_calibration_frame",
        fake_materialize,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "calibrate_uk_hmrc_income",
        fake_calibrate,
    )

    restored = restore_uk_hmrc_income_family(
        dataset,
        spi_tab_path=donor_path,
        hmrc_ods_path=tmp_path / "hmrc.ods",
        certified_candidate=candidate,
    )

    assert calls == ["targets", "replace", "impute", "materialize", "calibrate"]
    assert restored.dataset.household_weight_kind is WeightKind.CALIBRATED
    assert restored.dataset.mass_log == (mass_record,)
    assert restored.distributional_mass_shares == {
        "gift_aid": 1.0,
        "charitable_investment_gifts": 1.0,
    }


def test_stage_transform_retains_last_restoration_evidence(
    monkeypatch, tmp_path
) -> None:
    dataset = _dataset()
    expected = SimpleNamespace(dataset=dataset)
    monkeypatch.setattr(
        hmrc_restoration,
        "restore_uk_hmrc_income_family",
        lambda *_args, **_kwargs: expected,
    )
    transform = UKHMRCIncomeStageTransform(
        spi_tab_path=tmp_path / "put2223uk.tab",
        hmrc_ods_path=tmp_path / "hmrc.ods",
        certified_candidate=_candidate_identity(tmp_path),
    )

    assert transform(dataset) is dataset
    assert transform.last_result is expected


def test_restoration_blocks_before_source_io_when_frs_crosswalk_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    candidate = _candidate_identity(tmp_path)
    dataset = _dataset_from_source(candidate.path)
    missing = SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS[0]
    dataset = dataset.with_tables(person=dataset.person.drop(columns=[missing]))
    source_called = False

    def should_not_read_sources(*_args, **_kwargs):
        nonlocal source_called
        source_called = True
        raise AssertionError("FRS crosswalk preflight must run before source I/O")

    monkeypatch.setattr(
        hmrc_restoration,
        "materialize_hmrc_spi_income_band_targets",
        should_not_read_sources,
    )
    with pytest.raises(ValueError, match="employment_income is not a like-for-like"):
        restore_uk_hmrc_income_family(
            dataset,
            spi_tab_path=tmp_path / "put2223uk.tab",
            hmrc_ods_path=tmp_path / "hmrc.ods",
            certified_candidate=candidate,
        )
    assert not source_called


def test_restoration_rejects_unbound_in_memory_dataset(tmp_path) -> None:
    candidate = _candidate_identity(tmp_path)

    with pytest.raises(ValueError, match="arbitrary in-memory dataset"):
        restore_uk_hmrc_income_family(
            _dataset(),
            spi_tab_path=tmp_path / "put2223uk.tab",
            hmrc_ods_path=tmp_path / "hmrc.ods",
            certified_candidate=candidate,
        )


def test_restoration_rejects_forged_candidate_identity(tmp_path) -> None:
    candidate = _candidate_identity(tmp_path, verified=False)

    with pytest.raises(ValueError, match="must come from verify_certified"):
        restore_uk_hmrc_income_family(
            _dataset_from_source(candidate.path),
            spi_tab_path=tmp_path / "put2223uk.tab",
            hmrc_ods_path=tmp_path / "hmrc.ods",
            certified_candidate=candidate,
        )


def test_restoration_rejects_dataset_loaded_from_different_h5(tmp_path) -> None:
    candidate = _candidate_identity(tmp_path)
    other_source = tmp_path / "different-base.h5"

    with pytest.raises(ValueError, match="dataset source does not match"):
        restore_uk_hmrc_income_family(
            _dataset_from_source(other_source),
            spi_tab_path=tmp_path / "put2223uk.tab",
            hmrc_ods_path=tmp_path / "hmrc.ods",
            certified_candidate=candidate,
        )


def test_restoration_rejects_candidate_replaced_between_verify_and_load(
    monkeypatch,
    tmp_path,
) -> None:
    candidate_path = tmp_path / CERTIFIED_UK_CANDIDATE_FILENAME
    write_uk_national_dataset(_dataset(), candidate_path)
    monkeypatch.setattr(
        hmrc_restoration,
        "CERTIFIED_UK_CANDIDATE_SIZE_BYTES",
        candidate_path.stat().st_size,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "CERTIFIED_UK_CANDIDATE_SHA256",
        hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "assert_uk_hmrc_income_source_contract_current",
        lambda: None,
    )
    identity = verify_certified_uk_candidate(candidate_path)

    replacement = _dataset().with_tables(
        person=_dataset().person.assign(gift_aid=1.0),
    )
    write_uk_national_dataset(replacement, candidate_path)
    loaded_replacement = load_uk_national_dataset(candidate_path)

    with pytest.raises(ValueError, match="changed after SHA-256 verification"):
        restore_uk_hmrc_income_family(
            loaded_replacement,
            spi_tab_path=tmp_path / "put2223uk.tab",
            hmrc_ods_path=tmp_path / "hmrc.ods",
            certified_candidate=identity,
        )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("donor_sample_size", None),
        ("spi_prior_mass_share", 0.25),
        ("max_weight_ratio", 10.0),
        ("maximum_abs_relative_error", 0.10),
    ],
)
def test_restoration_rejects_unreviewed_release_parameter_overrides(
    tmp_path,
    override,
    value,
) -> None:
    candidate = _candidate_identity(tmp_path)
    kwargs = {
        "spi_tab_path": tmp_path / "put2223uk.tab",
        "hmrc_ods_path": tmp_path / "hmrc.ods",
        "certified_candidate": candidate,
        override: value,
    }

    with pytest.raises(ValueError, match="reviewed source manifest"):
        restore_uk_hmrc_income_family(
            _dataset_from_source(candidate.path),
            **kwargs,
        )
