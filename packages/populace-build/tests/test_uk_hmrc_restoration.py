from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import FitWeightRecord
from populace.build.uk_runtime import hmrc_restoration
from populace.build.uk_runtime.frs_hmrc_leaves import (
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
)
from populace.build.uk_runtime.hmrc_income import (
    HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeSourceProvenance,
    HMRCIncomeTargetSet,
)
from populace.build.uk_runtime.hmrc_restoration import (
    CERTIFIED_UK_CANDIDATE_FILENAME,
    CERTIFIED_UK_CANDIDATE_REVISION,
    CERTIFIED_UK_CANDIDATE_SHA256,
    CERTIFIED_UK_CANDIDATE_SIZE_BYTES,
    CERTIFIED_UK_CANDIDATE_TIER,
    UKCertifiedCandidateIdentity,
    UKHMRCIncomeStageTransform,
    restore_uk_hmrc_income_family,
    verify_certified_uk_candidate,
)
from populace.build.uk_runtime.national_build import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from populace.build.uk_runtime.national_frame import (
    UKStagingProvenance,
    _UKSourceFileFingerprint,
    uk_household_weight_kind,
    uk_national_frame,
)
from populace.build.uk_runtime.release_input_coverage import (
    DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE,
)
from populace.build.uk_runtime.spi_income import UKSPIIncomeImputationResult
from populace.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UKSPISupportResult,
    support_channel_column,
)
from populace.frame import Frame, MassChangeRecord, WeightKind

_TEST_SOURCE_FINGERPRINT = _UKSourceFileFingerprint(1, 2, 3, 4, 5)
_FRS_SOURCE_EVIDENCE = {
    "source_vintage": "2023-24",
    "adult": {"raw_variable": "ADULT.INEARNS", "sha256": "a" * 64},
    "benefits": {"raw_variable": "BENEFITS.BENAMT", "sha256": "b" * 64},
}


def _dataset() -> Frame:
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_household_id": [1],
                "person_benunit_id": [1],
                "gift_aid": [0.0],
                "charitable_investment_gifts": [0.0],
                FRS_HMRC_PAY_COLUMN: [20_000.0],
                FRS_HMRC_UBISJA_COLUMN: [100.0],
                FRS_HMRC_INCPBEN_COLUMN: [0.0],
                FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: [50.0],
                FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: [0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame(
            {
                "household_id": [1],
                "household_weight": [10.0],
            }
        ),
        time_period=HMRC_SPI_BUILD_PERIOD,
    )


def _dataset_from_source(
    path: Path,
    *,
    fingerprint: _UKSourceFileFingerprint = _TEST_SOURCE_FINGERPRINT,
) -> tuple[Frame, UKStagingProvenance]:
    """Model the provenance record that only the national H5 loader returns."""

    return _dataset(), UKStagingProvenance(
        source_h5=path.resolve(),
        fingerprint=fingerprint,
    )


def _candidate_identity(
    tmp_path: Path,
    *,
    verified: bool = True,
) -> UKCertifiedCandidateIdentity:
    identity = UKCertifiedCandidateIdentity(
        path=(tmp_path / CERTIFIED_UK_CANDIDATE_FILENAME).resolve(),
        filename=CERTIFIED_UK_CANDIDATE_FILENAME,
        tier=CERTIFIED_UK_CANDIDATE_TIER,
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


def _source_targets(tmp_path: Path) -> HMRCIncomeTargetSet:
    source = HMRCIncomeSourceProvenance(
        local_path=(tmp_path / "hmrc.ods").resolve(),
        sha256="c" * 64,
        publication_url="https://www.gov.uk/government/statistics/income-tax-liabilities",
        ods_url="https://assets.publishing.service.gov.uk/hmrc.ods",
        source_vintage="2023-24",
        source_tax_year="2023-24",
        source_tax_year_start=2023,
        build_period=HMRC_SPI_BUILD_PERIOD,
        table_names=("Table_3_6", "Table_3_7"),
        size_bytes=166_693,
        mime_type="application/vnd.oasis.opendocument.spreadsheet",
    )
    targets: list[HMRCIncomeBandTargetRecord] = []
    upper_bounds = (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None)
    for lower_bound, upper_bound in zip(
        HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
        upper_bounds,
        strict=True,
    ):
        for component in HMRC_SPI_INCOME_COMPONENTS:
            for measure, unit in (("count", "people"), ("amount", "GBP")):
                targets.append(
                    HMRCIncomeBandTargetRecord(
                        name=(
                            f"hmrc/{component}_{measure}_income_band_"
                            f"{lower_bound}_to_{upper_bound or 'inf'}"
                        ),
                        component=component,
                        measure=measure,
                        unit=unit,
                        value=float(len(targets) + 1),
                        period=HMRC_SPI_BUILD_PERIOD,
                        total_income_lower_bound=lower_bound,
                        total_income_upper_bound=upper_bound,
                    )
                )
    assert len(targets) == HMRC_SPI_TARGET_RECORD_COUNT
    return HMRCIncomeTargetSet(source=source, targets=tuple(targets))


def _support_and_imputation(
    dataset: Frame,
    tmp_path: Path,
    *,
    household_weights: tuple[float, ...] = (5.0, 5.0),
    gift_aid: tuple[float, ...] | None = None,
    charitable_gifts: tuple[float, ...] | None = None,
    assessable_income_adjustment: float = 0.0,
) -> tuple[UKSPISupportResult, UKSPIIncomeImputationResult]:
    if len(household_weights) < 2:
        raise ValueError("A synthetic replay fixture needs at least one SPI row.")
    row_count = len(household_weights)
    person = pd.concat([dataset.person] * row_count, ignore_index=True)
    person["person_id"] = np.arange(1, row_count + 1)
    person["person_household_id"] = np.arange(1, row_count + 1)
    person["person_benunit_id"] = np.arange(1, row_count + 1)
    person[support_channel_column("person")] = [BASE_FRS_SUPPORT_CHANNEL] + [
        SPI_SYNTHETIC_SUPPORT_CHANNEL
    ] * (row_count - 1)
    if gift_aid is None:
        gift_aid = (0.0, *([10.0] * (row_count - 1)))
    if charitable_gifts is None:
        charitable_gifts = (0.0, *([5.0] * (row_count - 1)))
    person["gift_aid"] = gift_aid
    person["charitable_investment_gifts"] = charitable_gifts

    household = pd.DataFrame(
        {
            "household_id": np.arange(1, row_count + 1),
            "household_weight": household_weights,
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(1, row_count + 1)})
    mass_record = MassChangeRecord(
        entity="household",
        old_total=float(dataset.table("household")["household_weight"].sum()),
        new_total=float(sum(household_weights)),
        declared_factor=1.0,
        reason="reviewed test allocation to one positive-mass SPI channel",
    )
    support = UKSPISupportResult(
        person=person.copy(),
        benunit=benunit,
        household=household,
        id_multiplier=10,
        spi_household_ids=tuple(range(2, row_count + 1)),
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=(mass_record,),
        replaced_spi_households=row_count - 1,
        spi_prior_mass_share=0.5,
    )

    imputed_person = person.copy()
    spi_count = row_count - 1
    total_earned = np.arange(10.0, 10.0 + spi_count)
    total_investment = np.arange(5.0, 5.0 + spi_count)
    assessable = total_earned + total_investment
    assessable[-1] += assessable_income_adjustment
    imputed_person[SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN] = (
        np.nan,
        *total_earned,
    )
    imputed_person[SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN] = (
        np.nan,
        *total_investment,
    )
    imputed_person[HMRC_SPI_ASSESSABLE_INCOME_COLUMN] = (np.nan, *assessable)
    imputation = UKSPIIncomeImputationResult(
        person=imputed_person,
        fit_weight_records=(
            FitWeightRecord("uk_spi_2022_23_income", "design"),
            FitWeightRecord("uk_frs_only_spi_fill", "importance"),
        ),
        donor_path=(tmp_path / "put2223uk.tab").resolve(),
        donor_sha256="d" * 64,
        donor_size_bytes=141_323_762,
        donor_rows=100_000,
        stage2_training_rows=1,
        spi_prediction_rows=spi_count,
        reviewed_absent_stage2_outputs={
            "incapacity_benefit_reported": "reviewed absent"
        },
    )
    return support, imputation


def _install_replay_mocks(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Frame,
    tmp_path: Path,
    *,
    household_weights: tuple[float, ...] = (5.0, 5.0),
    gift_aid: tuple[float, ...] | None = None,
    charitable_gifts: tuple[float, ...] | None = None,
    assessable_income_adjustment: float = 0.0,
) -> tuple[
    list[str],
    UKSPISupportResult,
    UKSPIIncomeImputationResult,
    HMRCIncomeTargetSet,
]:
    support, imputation = _support_and_imputation(
        dataset,
        tmp_path,
        household_weights=household_weights,
        gift_aid=gift_aid,
        charitable_gifts=charitable_gifts,
        assessable_income_adjustment=assessable_income_adjustment,
    )
    targets = _source_targets(tmp_path)
    calls: list[str] = []
    donor_identity = object()
    ods_identity = object()
    actual_crosswalk = hmrc_restoration.assert_frs_hmrc_auxiliary_crosswalk_available
    actual_report_builder = hmrc_restoration.build_conservative_hmrc_replay_report

    def fake_contract() -> None:
        calls.append("contract")

    def fake_verify_donor(path: Path) -> object:
        assert Path(path).name == "put2223uk.tab"
        calls.append("donor_identity")
        return donor_identity

    def fake_verify_ods(path: Path) -> object:
        assert Path(path).name == "hmrc.ods"
        calls.append("ods_identity")
        return ods_identity

    def fake_crosswalk(person: pd.DataFrame) -> None:
        calls.append("frs_crosswalk")
        actual_crosswalk(person)

    def fake_targets(verified: object, *, build_period: str) -> HMRCIncomeTargetSet:
        assert verified is ods_identity
        assert build_period == HMRC_SPI_BUILD_PERIOD
        calls.append("targets")
        return targets

    def fake_replace(**kwargs: object) -> UKSPISupportResult:
        assert kwargs["person"] is dataset.person
        assert kwargs["input_weight_kind"] is WeightKind.DESIGN
        calls.append("replace")
        return support

    def fake_impute(
        actual_support: UKSPISupportResult,
        _path: Path,
        **kwargs: object,
    ) -> UKSPIIncomeImputationResult:
        assert actual_support is support
        assert kwargs["verified_donor"] is donor_identity
        calls.append("impute")
        return imputation

    def fake_report(*args: object, **kwargs: object):
        calls.append("report")
        return actual_report_builder(*args, **kwargs)

    def forbidden_calibration(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("The adjudicated replay must never calibrate.")

    monkeypatch.setattr(
        hmrc_restoration,
        "assert_uk_hmrc_income_source_contract_current",
        fake_contract,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "verify_spi_donor_identity",
        fake_verify_donor,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "verify_hmrc_spi_collated_ods",
        fake_verify_ods,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "assert_frs_hmrc_auxiliary_crosswalk_available",
        fake_crosswalk,
    )
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
        "build_conservative_hmrc_replay_report",
        fake_report,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "calibrate_uk_hmrc_income",
        forbidden_calibration,
        raising=False,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "materialize_uk_hmrc_calibration_frame",
        forbidden_calibration,
        raising=False,
    )
    return calls, support, imputation, targets


def _restore(
    frame: Frame,
    candidate: UKCertifiedCandidateIdentity,
    tmp_path: Path,
    *,
    provenance: UKStagingProvenance | None = None,
):
    return restore_uk_hmrc_income_family(
        frame,
        spi_tab_path=tmp_path / "put2223uk.tab",
        hmrc_ods_path=tmp_path / "hmrc.ods",
        certified_candidate=candidate,
        staging_provenance=provenance,
        frs_source_evidence=_FRS_SOURCE_EVIDENCE,
    )


def test_hmrc_stage_transform_exposes_last_fit_weight_records(tmp_path: Path) -> None:
    transform = UKHMRCIncomeStageTransform(
        spi_tab_path=tmp_path / "put2223uk.tab",
        hmrc_ods_path=tmp_path / "hmrc.ods",
        certified_candidate=_candidate_identity(tmp_path),
    )
    records = (
        FitWeightRecord("uk_spi_2022_23_income", "design"),
        FitWeightRecord("uk_frs_only_spi_fill", "importance"),
    )

    assert transform.fit_weight_records == ()
    transform.last_result = SimpleNamespace(
        imputation=SimpleNamespace(fit_weight_records=records)
    )
    assert transform.fit_weight_records == records


def test_certified_candidate_verification_binds_size_sha_and_stable_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    assert identity.tier == "frs"
    assert identity.size_bytes == len(contents)
    assert identity.sha256 == hashlib.sha256(contents).hexdigest()
    assert identity._source_file_fingerprint is not None

    tampered = bytes((contents[0] ^ 1,)) + contents[1:]
    candidate.write_bytes(tampered)
    with pytest.raises(ValueError, match="sha256 .* does not match"):
        verify_certified_uk_candidate(candidate)


def test_restoration_binds_loaded_candidate_bytes_before_source_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    candidate_path = tmp_path / CERTIFIED_UK_CANDIDATE_FILENAME
    write_uk_national_frame(_dataset(), candidate_path)
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

    base = _dataset()
    replacement = uk_national_frame(
        person=base.person.assign(gift_aid=1.0),
        benunit=base.table("benunit"),
        household=base.table("household"),
        time_period=HMRC_SPI_BUILD_PERIOD,
    )
    write_uk_national_frame(replacement, candidate_path)
    loaded_replacement, replacement_provenance = load_uk_national_frame(candidate_path)
    monkeypatch.setattr(
        hmrc_restoration,
        "verify_spi_donor_identity",
        lambda _path: pytest.fail("source I/O preceded candidate byte binding"),
    )

    with pytest.raises(ValueError, match="changed after SHA-256 verification"):
        _restore(
            loaded_replacement,
            identity,
            tmp_path,
            provenance=replacement_provenance,
        )


def test_restoration_rejects_forged_or_unbound_candidate_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        hmrc_restoration,
        "assert_uk_hmrc_income_source_contract_current",
        lambda: None,
    )
    forged = _candidate_identity(tmp_path, verified=False)
    forged_frame, forged_provenance = _dataset_from_source(forged.path)
    with pytest.raises(ValueError, match="must come from verify_certified"):
        _restore(forged_frame, forged, tmp_path, provenance=forged_provenance)

    verified = _candidate_identity(tmp_path)
    with pytest.raises(ValueError, match="loaded from the verified"):
        _restore(_dataset(), verified, tmp_path)

    object.__setattr__(verified, "tier", "public")
    bound_frame, bound_provenance = _dataset_from_source(verified.path)
    with pytest.raises(ValueError, match="base identity does not match"):
        _restore(bound_frame, verified, tmp_path, provenance=bound_provenance)


def test_source_pair_is_verified_before_parse_or_support_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _candidate_identity(tmp_path)
    dataset, provenance = _dataset_from_source(candidate.path)
    calls: list[str] = []
    monkeypatch.setattr(
        hmrc_restoration,
        "assert_uk_hmrc_income_source_contract_current",
        lambda: None,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "verify_spi_donor_identity",
        lambda _path: calls.append("donor_identity") or object(),
    )

    def reject_ods(_path: Path) -> object:
        calls.append("ods_identity")
        raise RuntimeError("reviewed ODS identity mismatch")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("source parsing/support ran before paired preflight")

    monkeypatch.setattr(
        hmrc_restoration,
        "verify_hmrc_spi_collated_ods",
        reject_ods,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "materialize_hmrc_spi_income_band_targets",
        forbidden,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "replace_uk_spi_support_tables",
        forbidden,
    )
    monkeypatch.setattr(
        hmrc_restoration,
        "impute_uk_spi_income_support",
        forbidden,
    )

    with pytest.raises(RuntimeError, match="reviewed ODS identity mismatch"):
        _restore(dataset, candidate, tmp_path, provenance=provenance)
    assert calls == ["donor_identity", "ods_identity"]


def test_restoration_runs_replay_without_calibration_and_emits_208_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _candidate_identity(tmp_path)
    dataset, provenance = _dataset_from_source(candidate.path)
    calls, support, imputation, targets = _install_replay_mocks(
        monkeypatch,
        dataset,
        tmp_path,
    )

    restored = _restore(dataset, candidate, tmp_path, provenance=provenance)

    assert calls == [
        "contract",
        "donor_identity",
        "ods_identity",
        "frs_crosswalk",
        "targets",
        "replace",
        "impute",
        "report",
    ]
    assert restored.support is support
    assert restored.imputation is imputation
    assert restored.source_targets is targets
    assert uk_household_weight_kind(restored.frame) is WeightKind.IMPORTANCE
    assert restored.frame.mass_log == support.mass_log
    assert restored.post_draw_identity_rows == 1
    assert restored.distributional_mass_shares == {
        "gift_aid": 0.5,
        "charitable_investment_gifts": 0.5,
    }
    assert len(restored.replay_report.facts) == HMRC_SPI_TARGET_RECORD_COUNT == 208
    assert restored.replay_report.summary["excluded_with_fence"] == 208
    assert restored.replay_report.summary["exact_pass"] == 0
    assert restored.replay_report.source_evidence["certified_candidate"]["tier"] == (
        "frs"
    )
    evidence = restored.evidence()
    assert evidence["calibration"] == {
        "performed": False,
        "reason": (
            "Complete FRS Total Income band assignment is unavailable; the 208 "
            "facts are reviewed exclusions rather than biased calibration "
            "constraints."
        ),
        "output_weight_kind": "importance",
    }
    assert evidence["post_draw_identity"]["exact"] is True


def test_post_draw_total_income_identity_is_exact_not_tolerance_based(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _candidate_identity(tmp_path)
    dataset, provenance = _dataset_from_source(candidate.path)
    _install_replay_mocks(
        monkeypatch,
        dataset,
        tmp_path,
        assessable_income_adjustment=np.spacing(15.0),
    )

    with pytest.raises(RuntimeError, match=r"must equal deterministic TEI \+ TII"):
        _restore(dataset, candidate, tmp_path, provenance=provenance)


@pytest.mark.parametrize(
    "thin_column",
    ("gift_aid", "charitable_investment_gifts"),
)
def test_distributional_inputs_must_reach_one_ppm_positive_spi_mass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    thin_column: str,
) -> None:
    candidate = _candidate_identity(tmp_path)
    dataset, provenance = _dataset_from_source(candidate.path)
    household_weights = (9.899991, 0.1, 0.000009)
    values = {
        "gift_aid": (0.0, 1.0, 0.0),
        "charitable_investment_gifts": (0.0, 1.0, 0.0),
    }
    values[thin_column] = (0.0, 0.0, 1.0)
    _install_replay_mocks(
        monkeypatch,
        dataset,
        tmp_path,
        household_weights=household_weights,
        gift_aid=values["gift_aid"],
        charitable_gifts=values["charitable_investment_gifts"],
    )

    with pytest.raises(RuntimeError, match="required effective-mass") as error:
        _restore(dataset, candidate, tmp_path, provenance=provenance)
    assert thin_column in str(error.value)
    assert "e-07" in str(error.value)


def test_distributional_one_ppm_floor_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _candidate_identity(tmp_path)
    dataset, provenance = _dataset_from_source(candidate.path)
    _install_replay_mocks(
        monkeypatch,
        dataset,
        tmp_path,
        household_weights=(9.99999, 0.00001),
        gift_aid=(0.0, 1.0),
        charitable_gifts=(0.0, 1.0),
    )

    restored = _restore(dataset, candidate, tmp_path, provenance=provenance)

    assert set(restored.distributional_mass_shares) == {
        "gift_aid",
        "charitable_investment_gifts",
    }
    for share in restored.distributional_mass_shares.values():
        assert share >= DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
        assert share == pytest.approx(DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE)


def test_stage_transform_requires_retained_leaf_stage_and_forwards_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    transform = UKHMRCIncomeStageTransform(
        spi_tab_path=tmp_path / "put2223uk.tab",
        hmrc_ods_path=tmp_path / "hmrc.ods",
        certified_candidate=_candidate_identity(tmp_path),
    )
    with pytest.raises(RuntimeError, match="retained-leaves stage"):
        transform(dataset)

    stale_result = SimpleNamespace(
        frame=_dataset(),
        evidence=lambda: _FRS_SOURCE_EVIDENCE,
    )
    transform.retained_leaves_transform = SimpleNamespace(last_result=stale_result)
    with pytest.raises(RuntimeError, match="not bound to the frame"):
        transform(dataset)

    retained_result = SimpleNamespace(
        frame=dataset,
        evidence=lambda: _FRS_SOURCE_EVIDENCE,
    )
    transform.retained_leaves_transform.last_result = retained_result
    expected = SimpleNamespace(frame=dataset)
    forwarded: dict[str, object] = {}

    def fake_restore(*_args: object, **kwargs: object):
        forwarded.update(kwargs)
        return expected

    monkeypatch.setattr(
        hmrc_restoration,
        "restore_uk_hmrc_income_family",
        fake_restore,
    )

    assert transform(dataset) is dataset
    assert transform.last_result is expected
    assert forwarded["frs_source_evidence"] == _FRS_SOURCE_EVIDENCE


def test_stage_transform_binding_is_single_use_and_asserts_descent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The provenance binding restores the retired carrier's descent fence.

    Binding couples the provenance to the exact loaded frame; the stage
    consumes the binding on use, so a stale binding can never fence a later
    run, and a pipeline whose first stage consumed a substituted frame
    fails closed even when a matching load once happened.
    """

    loaded = _dataset()
    provenance = UKStagingProvenance(
        source_h5=(tmp_path / "populace_uk_2023.h5").resolve(),
        fingerprint=_TEST_SOURCE_FINGERPRINT,
    )
    transform = UKHMRCIncomeStageTransform(
        spi_tab_path=tmp_path / "put2223uk.tab",
        hmrc_ods_path=tmp_path / "hmrc.ods",
        certified_candidate=_candidate_identity(tmp_path),
    )
    forwarded: list[object] = []
    monkeypatch.setattr(
        hmrc_restoration,
        "restore_uk_hmrc_income_family",
        lambda frame, **kwargs: (
            forwarded.append(kwargs["staging_provenance"]),
            SimpleNamespace(frame=frame),
        )[1],
    )

    # Descent violation: the pipeline's first stage consumed a frame other
    # than the one the driver loaded and bound.
    substituted = _dataset()
    transform.retained_leaves_transform = SimpleNamespace(
        last_result=SimpleNamespace(
            frame=loaded, evidence=lambda: _FRS_SOURCE_EVIDENCE
        ),
        last_input=substituted,
    )
    transform.bind_staging_provenance(provenance, loaded)
    with pytest.raises(RuntimeError, match="did not start from the frame"):
        transform(loaded)
    assert transform.staging_provenance is None
    assert transform.bound_frame is None

    # Descent-consistent run forwards the bound provenance exactly once...
    transform.retained_leaves_transform.last_input = loaded
    transform.bind_staging_provenance(provenance, loaded)
    assert transform(loaded) is loaded
    assert forwarded == [provenance]

    # ...and a second run without rebinding gets no provenance (the real
    # restore then fails closed on staging_provenance=None).
    assert transform(loaded) is loaded
    assert forwarded == [provenance, None]


@pytest.mark.parametrize(
    ("override", "value"),
    (("donor_sample_size", None), ("spi_prior_mass_share", 0.25)),
)
def test_restoration_rejects_unreviewed_release_parameter_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: str,
    value: object,
) -> None:
    candidate = _candidate_identity(tmp_path)
    monkeypatch.setattr(
        hmrc_restoration,
        "assert_uk_hmrc_income_source_contract_current",
        lambda: None,
    )
    kwargs = {
        "spi_tab_path": tmp_path / "put2223uk.tab",
        "hmrc_ods_path": tmp_path / "hmrc.ods",
        "certified_candidate": candidate,
        "frs_source_evidence": _FRS_SOURCE_EVIDENCE,
        override: value,
    }

    frame, provenance = _dataset_from_source(candidate.path)
    with pytest.raises(ValueError, match="reviewed source manifest"):
        restore_uk_hmrc_income_family(
            frame,
            staging_provenance=provenance,
            **kwargs,
        )
