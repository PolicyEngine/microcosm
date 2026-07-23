"""Exact CPS ORG, occupation, and FLSA overtime stage contracts."""

from __future__ import annotations

import gzip
import hashlib

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceStageSpec
from populace.build.us_runtime import (
    ORG_2024_DONOR_CONTENT_SHA256,
    ORG_PREDICTORS,
    US_ORG_WAGES_OUTPUT_COLUMNS,
    derive_flsa_overtime_premium,
    derive_us_org_occupation_inputs,
    fetch_org_2024_donor,
    load_org_2024_donor,
    us_org_wages_signal_gate,
    us_org_wages_stage_spec,
)
from populace.build.us_runtime import org_wages as module
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _person(n: int = 1_000) -> pd.DataFrame:
    household = np.arange(1, n + 1, dtype=np.int64)
    occupation = np.full(n, 20, dtype=np.int16)
    occupation[:180] = 0
    occupation[180:420] = 53
    occupation[420:423] = 52
    occupation[423:443] = 8
    occupation[443:447] = 41
    occupation[447:737] = 1
    employment = np.zeros(n)
    employment[400:] = 52_000.0
    return pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype=np.int64),
            "person_household_id": household,
            "person_tax_unit_id": household + 10_000,
            "person_spm_unit_id": household + 20_000,
            "person_family_id": household + 30_000,
            "person_marital_unit_id": household + 40_000,
            "age": np.resize(np.arange(18, 78), n),
            "is_female": np.arange(n) % 2 == 0,
            "PRDTRACE": np.resize(np.asarray([1, 1, 2, 4]), n),
            "PRDTHSP": np.arange(n) % 7 == 0,
            "POCCU2": occupation,
            "employment_income_before_lsr": employment,
            "self_employment_income_before_lsr": 0.0,
            "weekly_hours_worked_before_lsr": np.where(employment > 0, 40.0, 0.0),
            "hours_worked_last_week": np.where(
                np.arange(n) >= 950, 50.0, np.where(employment > 0, 40.0, 0.0)
            ),
            "weeks_worked": np.where(employment > 0, 52.0, 0.0),
        }
    )


def _frame(person: pd.DataFrame) -> Frame:
    household_ids = person["person_household_id"].to_numpy()
    n = len(person)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": household_ids, "state_fips": np.resize([6, 36], n)}
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": person["person_tax_unit_id"].to_numpy()}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": person["person_spm_unit_id"].to_numpy()}
            ),
            "family": pd.DataFrame(
                {"family_id": person["person_family_id"].to_numpy()}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )


def _donor(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "employment_income": rng.uniform(10_000, 120_000, n),
            "weekly_hours_worked": rng.uniform(20, 60, n),
            "age": rng.integers(18, 80, n),
            "is_female": rng.integers(0, 2, n),
            "is_hispanic": rng.integers(0, 2, n),
            "race_wbho": rng.integers(1, 5, n),
            "state_fips": np.resize([6, 36], n),
            "hourly_wage": rng.uniform(8, 80, n),
            "is_paid_hourly": rng.integers(0, 2, n),
            "sample_weight": rng.uniform(1, 5, n),
        }
    )


def test_stage_spec_declares_complete_family_and_exact_predictors() -> None:
    spec = us_org_wages_stage_spec()
    assert isinstance(spec, SourceStageSpec)
    assert spec.stage == "org_wages"
    assert set(US_ORG_WAGES_OUTPUT_COLUMNS) <= set(spec.outputs)
    assert ORG_PREDICTORS == (
        "employment_income",
        "weekly_hours_worked",
        "age",
        "is_female",
        "is_hispanic",
        "race_wbho",
        "state_fips",
    )
    assert len(ORG_2024_DONOR_CONTENT_SHA256) == 64


def test_donor_loader_verifies_canonical_uncompressed_sha(tmp_path) -> None:
    donor = _donor(20)
    content = donor.to_csv(index=False).encode()
    path = tmp_path / "census_cps_org_2024_wages.csv.gz"
    path.write_bytes(gzip.compress(content, mtime=0))
    digest = hashlib.sha256(content).hexdigest()

    loaded = load_org_2024_donor(path, expected_content_sha256=digest)
    assert len(loaded) == 20
    with pytest.raises(ValueError, match="sha-256 verification"):
        load_org_2024_donor(path, expected_content_sha256="0" * 64)


def test_fetch_reuses_only_a_canonical_hash_matching_cache(
    tmp_path, monkeypatch
) -> None:
    content = _donor(20).to_csv(index=False).encode()
    path = tmp_path / "census_cps_org_2024_wages.csv.gz"
    path.write_bytes(gzip.compress(content, mtime=0))
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        module,
        "_load_month_from_network",
        lambda month: pytest.fail("valid cache must not fetch a monthly file"),
    )

    assert fetch_org_2024_donor(tmp_path, expected_content_sha256=digest) == path


def test_month_transform_matches_retired_raw_cps_fields() -> None:
    raw = pd.DataFrame(
        {
            "HRMIS": [4, 8, 4, 3],
            "gestfips": [6, 36, 12, 6],
            "prtage": [30, 45, 28, 32],
            "pesex": [1, 2, 2, 1],
            "ptdtrace": [1, 2, 3, 1],
            "pehspnon": [2, 2, 1, 2],
            "pworwgt": [100.0, 200.0, 150.0, 0.0],
            "pternwa": [100000.0, 80000.0, 120000.0, 90000.0],
            "pternhly": [2500.0, -1.0, 3000.0, 2000.0],
            "peernhry": [1, 2, 1, 1],
            "pehruslt": [40.0, 40.0, 50.0, 40.0],
            "prerelg": [1, 1, 1, 1],
            "pemlr": [1, 1, 2, 1],
            "peio1cow": [1, 4, 2, 1],
        }
    )

    transformed = module._transform_month(raw)

    assert len(transformed) == 3
    assert transformed["hourly_wage"].tolist() == [25.0, 20.0, 30.0]
    assert transformed["is_paid_hourly"].tolist() == [1.0, 0.0, 1.0]
    assert transformed["employment_income"].tolist() == [52_000.0, 41_600.0, 62_400.0]
    assert transformed.dtypes.eq(np.dtype("float32")).all()


def test_occupation_carries_match_retired_poccu2_codes() -> None:
    person = pd.DataFrame(
        {
            "PRDTRACE": [1, 2, 4, 1, 1, 1, 1],
            "PRDTHSP": [0, 1, 0, 0, 0, 0, 0],
            "POCCU2": [53, 52, 8, 41, 1, 50, 20],
        }
    )
    result = derive_us_org_occupation_inputs(person)

    assert result["cps_race"].tolist() == [1, 2, 4, 1, 1, 1, 1]
    assert result["is_hispanic"].tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result["has_never_worked"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result["is_military"].tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result["is_computer_scientist"].tolist()[2]
    assert result["is_farmer_fisher"].tolist()[3]
    assert result["is_executive_administrative_professional"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_flsa_proxy_uses_annual_wage_share_and_exemption_screen() -> None:
    # usual hours at/below the threshold: the reference-week leg reproduces the
    # retired annual-wage-share arithmetic exactly (the occasional-overtime
    # snapshot convention is retained, not re-derived).
    policy = (107_432.0, 35_568.0, 57_470.4, 40.0, 1.5)
    premium = derive_flsa_overtime_premium(
        time_period=2024,
        employment_income=np.asarray([57_200, 60_000, 60_000, 100_000, 50_000]),
        hours_worked_last_week=np.asarray([50, 50, 50, 50, 50]),
        usual_weekly_hours=np.asarray([40, 40, 40, 40, 40]),
        weeks_worked=np.asarray([52, 52, 52, 52, 52]),
        is_paid_hourly=np.asarray([True, False, False, False, True]),
        has_never_worked=np.asarray([False, False, False, False, True]),
        is_military=np.zeros(5, dtype=bool),
        is_executive_administrative_professional=np.asarray(
            [False, True, False, False, False]
        ),
        is_farmer_fisher=np.zeros(5, dtype=bool),
        is_computer_scientist=np.asarray([False, False, True, False, False]),
        policy=policy,
    )
    np.testing.assert_allclose(premium, [5_200, 0, 0, 100_000 / 11, 0], rtol=1e-6)


def test_flsa_two_signal_estimator_usual_hours_leg() -> None:
    """Usual weekly hours above the threshold carry the persistent-overtime leg.

    share(45) = 0.5*5 / (40 + 1.5*5) = 1/19; share(50) = 0.5*10 / (40 + 15)
    = 1/11. Workers whose usual week exceeds the threshold are carriers even
    when the (out-of-income-year) reference week is at or below it, and their
    annualizer is the persistent usual-hours share, never a single hot week.
    """
    policy = (107_432.0, 35_568.0, 57_470.4, 40.0, 1.5)
    premium = derive_flsa_overtime_premium(
        time_period=2024,
        employment_income=np.asarray(
            [57_000, 57_000, 55_000, 55_000, 60_000, 57_000, 57_000, 120_000]
        ),
        hours_worked_last_week=np.asarray([40, 60, 0, 50, 40, 40, 40, 40]),
        usual_weekly_hours=np.asarray([45, 45, 50, 0, 50, 45, 45, 45]),
        weeks_worked=np.asarray([52, 52, 52, 52, 52, 0, 52, 52]),
        is_paid_hourly=np.asarray(
            [False, False, False, False, False, False, False, True]
        ),
        has_never_worked=np.asarray(
            [False, False, False, False, False, False, True, False]
        ),
        is_military=np.zeros(8, dtype=bool),
        is_executive_administrative_professional=np.asarray(
            [False, False, False, False, True, False, False, False]
        ),
        is_farmer_fisher=np.zeros(8, dtype=bool),
        is_computer_scientist=np.zeros(8, dtype=bool),
        policy=policy,
    )
    np.testing.assert_allclose(
        premium,
        [
            3_000,  # usual 45, reference week 40: recovered carrier, share 1/19
            3_000,  # usual 45, reference week 60: persistent signal wins
            5_000,  # usual 50, reference week 0 (absent): recovered, share 1/11
            5_000,  # usual 0 record, reference 50: reference-week leg, share 1/11
            0,  # salaried EAP above the salary-basis threshold stays exempt
            0,  # zero weeks worked stays zero
            0,  # always-exempt (never worked) stays zero
            120_000 / 19,  # hourly worker above the HCE threshold stays covered
        ],
        rtol=1e-6,
    )


def test_imputation_zeroes_inactive_and_union_is_deterministic(monkeypatch) -> None:
    class Fitted:
        def predict(self, features):
            return pd.DataFrame(
                {
                    "hourly_wage": np.full(len(features), 25.0),
                    "is_paid_hourly": np.resize([0.9, 0.1], len(features)),
                },
                index=features.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return Fitted()

    monkeypatch.setattr("populace.fit.QRF", FakeQRF)
    frame = _frame(_person(100))
    first_carried, first = module.impute_us_org_wages(frame, _donor(), seed=7)
    second_carried, second = module.impute_us_org_wages(frame, _donor(), seed=99)

    inactive = frame.table("person")["employment_income_before_lsr"].to_numpy() <= 0
    assert (first.loc[inactive, "hourly_wage"] == 0).all()
    assert not first.loc[inactive, "is_paid_hourly"].any()
    assert not first.loc[inactive, "is_union_member_or_covered"].any()
    np.testing.assert_array_equal(
        first["is_union_member_or_covered"], second["is_union_member_or_covered"]
    )
    pd.testing.assert_frame_equal(first_carried, second_carried)


def test_union_assignment_matches_unweighted_state_quotas() -> None:
    n = 201
    features = pd.DataFrame(
        {
            "employment_income": np.r_[np.full(200, 50_000.0), 0.0],
            "weekly_hours_worked": np.full(n, 40.0),
            "age": np.resize(np.arange(20, 70), n),
            "is_female": np.arange(n) % 2,
            "is_hispanic": np.arange(n) % 7 == 0,
            "race_wbho": np.resize([1, 2, 3, 4], n),
            "state_fips": np.r_[np.full(100, 6), np.full(100, 37), 6],
        }
    )

    assigned = module._assign_union(features)

    assert assigned[:100].sum() == 16  # California 16.3%, rounded.
    assert assigned[100:200].sum() == 3  # North Carolina 3.1%, rounded.
    assert not assigned[-1]  # inactive rows never receive union coverage.


def test_live_2024_flsa_policy_values_when_us_extra_is_installed() -> None:
    pytest.importorskip("policyengine_us")
    assert module._flsa_policy(2024) == pytest.approx(
        (107_432.0, 35_568.0, 57_470.4, 40.0, 1.5)
    )


def _plausible_surface() -> Frame:
    person = _person()
    person["cps_race"] = person["PRDTRACE"]
    person["is_hispanic"] = person["PRDTHSP"].ne(0)
    carried = derive_us_org_occupation_inputs(person)
    for column in carried:
        person[column] = carried[column]
    person["hourly_wage"] = 0.0
    person.loc[400:949, "hourly_wage"] = 25.0
    person["is_paid_hourly"] = False
    person.loc[400:649, "is_paid_hourly"] = True
    person["is_union_member_or_covered"] = False
    person.loc[700:759, "is_union_member_or_covered"] = True
    person["fsla_overtime_premium"] = 0.0
    # These 50 are non-exempt code-20 workers with 50 hours and positive wages.
    person.loc[950:999, "fsla_overtime_premium"] = 52_000 / 11
    # These 50 carry the usual-hours leg: reference week at the threshold but a
    # 45-hour usual week (share 1/19) — valid carriers under the two-signal
    # estimator.
    person.loc[900:949, "weekly_hours_worked_before_lsr"] = 45.0
    person.loc[900:949, "fsla_overtime_premium"] = 52_000 / 19
    return _frame(person)


def test_signal_gate_passes_plausible_coherent_surface() -> None:
    gate = us_org_wages_signal_gate(_plausible_surface())
    assert gate.passed, gate.failures


def test_signal_gate_rejects_zero_and_structurally_impossible_premium() -> None:
    frame = _plausible_surface()
    frame.table("person")["fsla_overtime_premium"] = 0.0
    gate = us_org_wages_signal_gate(frame)
    assert not gate.passed
    assert any("fsla_overtime_premium" in failure for failure in gate.failures)

    impossible = _plausible_surface()
    impossible.table("person").loc[500, "fsla_overtime_premium"] = 1_000.0
    impossible.table("person").loc[500, "hours_worked_last_week"] = 40.0
    impossible.table("person").loc[500, "weekly_hours_worked_before_lsr"] = 40.0
    gate = us_org_wages_signal_gate(impossible)
    assert not gate.passed
    assert any("positive_without_overtime" in failure for failure in gate.failures)


def test_signal_gate_accepts_usual_hours_leg_carriers() -> None:
    # A positive premium with a 40-hour reference week is coherent when the
    # usual week exceeds the threshold; only both signals at or below the
    # threshold make a positive premium structurally impossible.
    frame = _plausible_surface()
    person = frame.table("person")
    assert (person.loc[900:949, "hours_worked_last_week"] == 40.0).all()
    assert (person.loc[900:949, "weekly_hours_worked_before_lsr"] == 45.0).all()
    assert (person.loc[900:949, "fsla_overtime_premium"] > 0).all()
    gate = us_org_wages_signal_gate(frame)
    assert gate.passed, gate.failures


def test_stage_derives_usual_hours_leg_end_to_end(monkeypatch) -> None:
    pytest.importorskip("policyengine_us")

    class Fitted:
        def predict(self, features):
            return pd.DataFrame(
                {
                    "hourly_wage": np.full(len(features), 25.0),
                    "is_paid_hourly": np.ones(len(features)),
                },
                index=features.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return Fitted()

    monkeypatch.setattr("populace.fit.QRF", FakeQRF)
    person = _person(100)
    # Recovered carrier: usual 45-hour week, 40-hour reference week.
    person.loc[50, "employment_income_before_lsr"] = 57_000.0
    person.loc[50, "weekly_hours_worked_before_lsr"] = 45.0
    person.loc[50, "hours_worked_last_week"] = 40.0
    person.loc[50, "weeks_worked"] = 52.0
    frame = module.with_us_org_wages_inputs(
        _frame(person), seed=0, time_period=2024, org_donor=_donor()
    )
    premium = frame.table("person")["fsla_overtime_premium"]
    np.testing.assert_allclose(premium.iloc[50], 3_000.0, rtol=1e-5)
    inactive = person["employment_income_before_lsr"].to_numpy() <= 0
    assert (premium.to_numpy()[inactive] == 0).all()
