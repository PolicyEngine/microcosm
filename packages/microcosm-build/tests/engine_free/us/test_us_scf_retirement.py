"""Observed SCF joint contribution tuples and source-boundary invariants."""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import scf_retirement as scf


def _person(**changes) -> dict[str, float]:
    row = dict.fromkeys(scf.SCF_RETIREMENT_SOURCE_COLUMNS, 0.0)
    row.update(
        y1=11,
        yy1=1,
        x42001=500,
        x8022=40,
        x8021=1,
        x4106=1,
        x4110=40,
        x4111=52,
    )
    row.update(changes)
    return row


def _ira(**changes) -> dict[str, float]:
    return _person(**dict(x6791=1, x6792=2_000, x6447=1, x6553=20_000) | changes)


def _employer(**changes) -> dict[str, float]:
    return _person(**dict(x4201=1, x11001=2, x11040=1, x11042=100, x11043=2) | changes)


def test_source_participation_and_amounts_are_joint_and_nonnegative():
    ira = _ira(y1=11)
    employer = _employer(y1=21)
    both = _ira(y1=31, x4201=1, x11001=2, x11040=1, x11042=100, x11043=2)
    raw = pd.DataFrame([ira, employer, both])
    donor = scf.extract_scf_retirement_donor(raw)
    values = donor[list(scf.SCF_RETIREMENT_OUTPUT_COLUMNS)].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    np.testing.assert_allclose(values.sum(axis=1), [2_000, 5_200, 7_200])
    i = values[:, 2:4].sum(axis=1) > 0
    e = values[:, [0, 1, 4]].sum(axis=1) > 0
    assert list(zip(i, e, strict=True)) == [(True, False), (False, True), (True, True)]
    np.testing.assert_allclose(donor.scf_weight, [100, 100, 100])
    # The existing employer tax-type proxy allocates within the observed plan.
    np.testing.assert_allclose(values[1, :2], [4_420, 780])


def test_balances_split_only_observed_ira_pool_and_keep_its_total():
    raw = pd.DataFrame(
        [
            _person(
                x6791=1,
                x6792=6_000,
                x6444=1,
                x6446=1,
                x6447=1,
                x6551=10_000,
                x6552=5_000,
                x6553=15_000,
            )
        ]
    )
    donor = scf.extract_scf_retirement_donor(raw)
    np.testing.assert_allclose(
        donor[list(scf.SCF_RETIREMENT_OUTPUT_COLUMNS)].iloc[0],
        [0, 0, 4_000, 2_000, 0],
    )
    assert donor.total_contributions.iloc[0] == 6_000


def test_account_ownership_without_contributions_never_creates_ira_participation():
    employer = _employer(x6447=1, x6553=500_000, x6791=5)
    donor = scf.extract_scf_retirement_donor(pd.DataFrame([employer]))
    assert donor.traditional_ira_contributions_desired.iloc[0] == 0
    assert donor.roth_ira_contributions_desired.iloc[0] == 0


def test_spouse_is_a_separate_person_and_se_amounts_keep_joint_ira():
    raw = pd.DataFrame(
        [
            _person(
                x104=37,
                x103=2,
                x4706=2,
                x4710=30,
                x4711=50,
                x4801=1,
                x11301=21,
                x11340=1,
                x11342=2,
                x11343=18,
                x6793=1,
                x6794=1_000,
                x6448=1,
                x6559=10_000,
            )
        ]
    )
    donor = scf.extract_scf_retirement_donor(raw)
    assert len(donor) == 1
    assert donor.donor_id.iloc[0].endswith(":1")
    assert donor.is_female.iloc[0]
    assert donor.has_self_employment.iloc[0]
    np.testing.assert_allclose(
        donor[list(scf.SCF_RETIREMENT_OUTPUT_COLUMNS)].iloc[0], [0, 0, 0, 1_000, 3_000]
    )


def test_defined_benefit_employee_contributions_are_deferred_to_a2():
    raw = pd.DataFrame([_ira(x4201=1, x11001=1, x11040=1, x11042=100, x11043=2)])
    donor = scf.extract_scf_retirement_donor(raw)
    assert donor.total_contributions.iloc[0] == 2_000
    assert donor.traditional_401k_contributions_desired.iloc[0] == 0


@pytest.mark.parametrize(
    "changes", [dict(x11043=14), dict(x4201=3), dict(x11001=-7), dict(x11042=-2)]
)
def test_unusable_workplace_observation_excludes_whole_tuple(changes):
    good = _ira(y1=11)
    bad = _ira(y1=21, x4201=1, x11001=2, x11040=1, x11042=100, x11043=2)
    bad.update(changes)
    donor = scf.extract_scf_retirement_donor(pd.DataFrame([good, bad]))
    assert list(donor.donor_id) == ["11:1:0"]
    assert donor.attrs["scf_retirement"]["excluded_person_implicates"] == 1


def test_stopped_or_variable_contributions_use_observed_prior_year_amount():
    raw = pd.DataFrame([_employer(x11040=3, x11042=0, x11043=0, x11045=200, x11046=31)])
    donor = scf.extract_scf_retirement_donor(raw)
    assert donor.total_contributions.iloc[0] == 4_800
    assert donor.attrs["scf_retirement"]["ira_contribution_year"] == 2021


def test_ambiguous_zero_balance_ira_mix_is_not_guessed():
    good = _ira(y1=11)
    bad = _person(y1=21, x6791=1, x6792=2_000, x6444=1, x6447=1)
    donor = scf.extract_scf_retirement_donor(pd.DataFrame([good, bad]))
    assert len(donor) == 1


def test_mixed_ira_keogh_contributors_are_excluded_but_pure_keogh_retained():
    mixed = _ira(y1=11, x3605=1, x6554=20_000)
    pure = _person(y1=21, x6791=1, x6792=5_000, x3605=1, x6554=20_000)
    donor = scf.extract_scf_retirement_donor(pd.DataFrame([mixed, pure]))
    assert list(donor.donor_id) == ["21:1:0"]
    assert donor.self_employed_pension_contributions_desired.iloc[0] == 5_000
    assert donor.traditional_ira_contributions_desired.iloc[0] == 0
    assert donor.attrs["scf_retirement"]["excluded_person_implicates"] == 1


def test_duplicate_source_identity_and_invalid_roth_share_refuse():
    raw = pd.DataFrame([_ira(), _ira()])
    with pytest.raises(ValueError, match="unique"):
        scf.extract_scf_retirement_donor(raw)
    with pytest.raises(ValueError, match="between zero and one"):
        scf.extract_scf_retirement_donor(raw, roth_dc_share=1.1)


def test_fetch_checks_archive_and_member_bytes(tmp_path, monkeypatch):
    member = b"public-SCF-fixture"
    archive_io = io.BytesIO()
    with zipfile.ZipFile(archive_io, "w") as archive:
        archive.writestr(scf.SCF_RETIREMENT_MEMBER, member)
    archive_bytes = archive_io.getvalue()
    monkeypatch.setattr(
        scf, "SCF_RETIREMENT_ZIP_SHA256", hashlib.sha256(archive_bytes).hexdigest()
    )
    monkeypatch.setattr(
        scf, "SCF_RETIREMENT_MEMBER_SHA256", hashlib.sha256(member).hexdigest()
    )
    monkeypatch.setattr(
        scf.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(archive_bytes)
    )
    target = scf.fetch_scf_retirement_extract(tmp_path)
    assert target.read_bytes() == member
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Cached.*mismatch"):
        scf.fetch_scf_retirement_extract(tmp_path)
