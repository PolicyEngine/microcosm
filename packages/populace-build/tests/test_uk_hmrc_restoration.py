from __future__ import annotations

import hashlib
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
    UKHMRCIncomeStageTransform,
    restore_uk_hmrc_income_family,
    verify_certified_uk_candidate,
)
from populace.build.uk_runtime.national_build import UKNationalDataset
from populace.build.uk_runtime.spi_income import UKSPIIncomeImputationResult
from populace.build.uk_runtime.spi_support import UKSPISupportResult
from populace.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
)


def _dataset() -> UKNationalDataset:
    return UKNationalDataset(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_household_id": [1],
                "person_benunit_id": [1],
                "gift_aid": [0.0],
                "charitable_investment_gifts": [0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [10.0],
            }
        ),
        time_period="2023",
    )


def test_certified_candidate_verification_checks_size_and_sha(
    monkeypatch,
    tmp_path,
) -> None:
    candidate = tmp_path / "candidate.h5"
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
    dataset = _dataset()
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
            "person": pd.DataFrame(
                {"person_id": [1], "person_household_id": [1]}
            ),
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
    )

    assert calls == ["targets", "replace", "impute", "materialize", "calibrate"]
    assert restored.dataset.household_weight_kind is WeightKind.CALIBRATED
    assert restored.dataset.mass_log == (mass_record,)
    assert restored.distributional_mass_shares == {
        "gift_aid": 1.0,
        "charitable_investment_gifts": 1.0,
    }


def test_stage_transform_retains_last_restoration_evidence(monkeypatch, tmp_path) -> None:
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
    )

    assert transform(dataset) is dataset
    assert transform.last_result is expected
