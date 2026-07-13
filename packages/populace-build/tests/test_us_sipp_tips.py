"""US SIPP tip-income and tipped-occupation source-stage tests.

The retired eCPS pipeline annualized observed-source December SIPP monthly tip
amounts while excluding Census allocation-flag columns from the dollar sum,
and QRF-imputed the resulting person-level income with Treasury-listed
occupation as a predictor rather than a domain mask.
These tests pin that source transformation and the release-stage healing and
signal contracts.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceStageSpec
from populace.build.us_runtime import (
    SIPP_2023_TIP_DONOR_REVISION,
    SIPP_2023_TIP_DONOR_SHA256,
    SIPP_2023_TIP_DONOR_URL,
    SIPP_TIP_OUTPUT_COLUMNS,
    SIPP_TIP_PREDICTORS,
    derive_treasury_tipped_occupation_code,
    fetch_sipp_2023_tip_donor,
    impute_us_sipp_tips,
    load_sipp_2023_tip_donor,
    us_sipp_tips_signal_gate,
    us_sipp_tips_stage_spec,
    us_sipp_tips_summary,
    with_us_sipp_tip_inputs,
)
from populace.build.us_runtime import sipp_tips as module
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024

_DONOR_WEIGHT_COLUMN = "sipp_weight"
_EXPECTED_DONOR_SHA256 = (
    "1f0bcb8e045ef1118e8eba4b4a2997bdaaf947bd0dd09d41fa7c7d5657a3d7d5"
)


def _raw_sipp_row(
    *,
    ssuid: int,
    month: int,
    age: int,
    monthly_income: float,
    tips: tuple[float, ...] = (),
    occupations: tuple[int, ...] = (),
    allocation_flags: tuple[int, ...] = (),
    weight: float = 1.0,
) -> dict[str, float | int]:
    """Return one ``pu2023_slim.csv``-shaped synthetic person-month."""

    row: dict[str, float | int] = {
        "SSUID": ssuid,
        "MONTHCODE": month,
        "WPFINWGT": weight,
        "TAGE": age,
        "TPTOTINC": monthly_income,
        # A distractor the retired broad ``contains('TXAMT')`` selector would
        # have summed. The port must read only the seven TJB amount fields.
        "SOME_TXAMT_OTHER": 50_000.0,
    }
    for job in range(1, 8):
        position = job - 1
        row[f"TJB{job}_TXAMT"] = tips[position] if position < len(tips) else 0.0
        row[f"AJB{job}_TXAMT"] = (
            allocation_flags[position] if position < len(allocation_flags) else 0
        )
        row[f"TJB{job}_OCC"] = (
            occupations[position] if position < len(occupations) else 9999
        )
    return row


def _raw_sipp() -> pd.DataFrame:
    """Small panel with December, non-December, and allocated records."""

    return pd.DataFrame(
        [
            # The same adult's November record must not enter the annual donor.
            _raw_sipp_row(
                ssuid=1,
                month=11,
                age=35,
                monthly_income=2_000.0,
                tips=(9_999.0,),
                occupations=(4040,),
            ),
            # $100 + $50 per month -> exactly $1,800 annually.
            _raw_sipp_row(
                ssuid=1,
                month=12,
                age=35,
                monthly_income=2_000.0,
                tips=(100.0, 50.0),
                occupations=(4040, 9999),
                weight=2.0,
            ),
            # A child in the adult's December household pins composition counts.
            _raw_sipp_row(
                ssuid=1,
                month=12,
                age=5,
                monthly_income=0.0,
            ),
            # This allocated child is not a training target, but still counts
            # in the retained adult's December household composition.
            _raw_sipp_row(
                ssuid=1,
                month=12,
                age=3,
                monthly_income=0.0,
                allocation_flags=(2,),
            ),
            # SIPP status 2 is an allocated/imputed source amount. It must be
            # excluded from training, not added to the dollar amount. Statuses
            # 0, 1, and 9 are observed/derivable in the retired source-quality
            # contract.
            _raw_sipp_row(
                ssuid=2,
                month=12,
                age=40,
                monthly_income=4_000.0,
                tips=(25_000.0,),
                occupations=(4110,),
                allocation_flags=(2,),
            ),
        ]
    )


def _donor_table(n: int = 180) -> pd.DataFrame:
    """Fast, varied donor table for QRF plumbing tests."""

    rng = np.random.default_rng(1)
    tipped = np.arange(n) % 3 == 0
    donor = pd.DataFrame(
        {
            "employment_income": rng.gamma(2.0, 18_000.0, n),
            "age": rng.integers(18, 80, n).astype(np.float64),
            "count_under_18": rng.integers(0, 4, n).astype(np.float64),
            "count_under_6": rng.integers(0, 2, n).astype(np.float64),
            "is_tipped_occupation": tipped.astype(np.float64),
            "tip_income": np.where(tipped, rng.gamma(2.0, 1_200.0, n) + 100.0, 0.0),
            "treasury_tipped_occupation_code": np.where(tipped, 101, 0),
            _DONOR_WEIGHT_COLUMN: rng.uniform(0.5, 3.0, n),
        }
    )
    return donor


def _person_rows(n: int = 120) -> pd.DataFrame:
    """ASEC-shaped recipients with a mix of listed and unlisted occupations."""

    rng = np.random.default_rng(2)
    tipped = np.arange(n) % 10 == 0
    return pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype=np.int64),
            "person_household_id": np.arange(1, n + 1, dtype=np.int64),
            "employment_income_before_lsr": rng.gamma(2.0, 18_000.0, n),
            "age": rng.integers(20, 70, n).astype(np.float64),
            "PEIOOCC": np.where(tipped, 4040, 9999),
        }
    )


def _us_frame(person: pd.DataFrame, *, weights: np.ndarray | None = None) -> Frame:
    person = person.copy()
    n = len(person)
    household_ids = person["person_household_id"].to_numpy(dtype=np.int64)
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = household_ids + 1_000
    person["person_spm_unit_id"] = household_ids + 2_000
    person["person_family_id"] = household_ids + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype=np.int64) + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unique_households + 2_000}),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype=np.int64) + 4_000}
        ),
    }
    household_weights = (
        np.ones(len(unique_households), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=household_weights,
                kind=WeightKind.DESIGN,
            )
        },
    )


def _frame_with_tip_surface(
    *,
    n: int = 1_000,
    tipped_count: int = 70,
    positive_tip_count: int = 8,
) -> Frame:
    person = _person_rows(n)
    codes = np.zeros(n, dtype=np.int16)
    codes[:tipped_count] = 101
    tips = np.zeros(n, dtype=np.float64)
    tips[:positive_tip_count] = np.arange(1, positive_tip_count + 1) * 250.0
    person["treasury_tipped_occupation_code"] = codes
    person["tip_income"] = tips
    return _us_frame(person)


# --------------------------------------------------------------------------- #
# Source declaration and pinned artifact
# --------------------------------------------------------------------------- #
def test_stage_spec_loads_and_declares_both_outputs() -> None:
    spec = us_sipp_tips_stage_spec()
    assert isinstance(spec, SourceStageSpec)
    assert spec.stage == "sipp_tips"
    assert SIPP_TIP_OUTPUT_COLUMNS == (
        "tip_income",
        "treasury_tipped_occupation_code",
    )
    assert set(SIPP_TIP_OUTPUT_COLUMNS) <= set(spec.outputs)
    assert len(spec.artifacts) == 1
    artifact = spec.artifacts[0]
    assert artifact["sha256"] == SIPP_2023_TIP_DONOR_SHA256
    assert SIPP_2023_TIP_DONOR_REVISION in artifact["locator"]
    tip_model = next(
        operation
        for operation in spec.operations
        if operation.kind == "fit_tip_income_model"
    )
    assert tuple(tip_model.parameters["predictors"]) == SIPP_TIP_PREDICTORS
    assert tuple(tip_model.parameters["source_columns"]) == (
        module._SIPP_TIP_AMOUNT_COLUMNS
    )
    assert tuple(tip_model.parameters["allocation_flag_columns"]) == (
        module._SIPP_TIP_ALLOCATION_COLUMNS
    )
    assert tip_model.parameters["observed_status_values"] == [0, 1, 9]
    assert not any(operation.kind == "zero_when_false" for operation in spec.operations)


def test_predictors_match_retired_sipp_tip_model() -> None:
    assert SIPP_TIP_PREDICTORS == (
        "employment_income",
        "age",
        "count_under_18",
        "count_under_6",
        "is_tipped_occupation",
    )


def test_sipp_donor_coordinate_and_sha_are_pinned() -> None:
    assert SIPP_2023_TIP_DONOR_URL.endswith("/pu2023_slim.csv")
    assert "/resolve/" in SIPP_2023_TIP_DONOR_URL
    assert SIPP_2023_TIP_DONOR_SHA256 == _EXPECTED_DONOR_SHA256


# --------------------------------------------------------------------------- #
# Donor loading and direct occupation derivation
# --------------------------------------------------------------------------- #
def test_load_donor_uses_december_exact_tip_sum_and_observed_rows(tmp_path) -> None:
    path = tmp_path / "pu2023_slim.csv"
    _raw_sipp().to_csv(path, index=False)

    donor = load_sipp_2023_tip_donor(path)

    # November and the allocated December record are excluded.
    assert sorted(donor["age"].tolist()) == [5.0, 35.0]
    adult = donor.loc[donor["age"] == 35].iloc[0]
    assert adult["tip_income"] == pytest.approx((100.0 + 50.0) * 12.0)
    assert adult["employment_income"] == pytest.approx(2_000.0 * 12.0)
    assert adult["treasury_tipped_occupation_code"] == 101
    assert adult["count_under_18"] == 2
    assert adult["count_under_6"] == 2
    assert adult[_DONOR_WEIGHT_COLUMN] == pytest.approx(2.0)


def test_load_donor_requires_allocation_flags(tmp_path) -> None:
    raw = _raw_sipp().drop(columns=["AJB7_TXAMT"])
    path = tmp_path / "pu2023_slim.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_sipp_2023_tip_donor(path)


def test_load_donor_verifies_requested_sha(tmp_path) -> None:
    path = tmp_path / "pu2023_slim.csv"
    _raw_sipp().to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    donor = load_sipp_2023_tip_donor(path, expected_sha256=digest)
    assert len(donor) == 2
    with pytest.raises(ValueError, match="sha-256 verification"):
        load_sipp_2023_tip_donor(path, expected_sha256="0" * 64)


def test_load_donor_retains_allowed_statuses_one_and_nine(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            _raw_sipp_row(
                ssuid=1,
                month=12,
                age=31,
                monthly_income=2_500.0,
                tips=(10.0,),
                allocation_flags=(1,),
            ),
            _raw_sipp_row(
                ssuid=2,
                month=12,
                age=42,
                monthly_income=3_500.0,
                tips=(20.0,),
                allocation_flags=(9,),
            ),
        ]
    )
    path = tmp_path / "pu2023_slim.csv"
    raw.to_csv(path, index=False)

    donor = load_sipp_2023_tip_donor(path)

    assert sorted(donor["age"].tolist()) == [31.0, 42.0]
    assert sorted(donor["tip_income"].tolist()) == [120.0, 240.0]


def test_treasury_tipped_occupation_mapping() -> None:
    derived = derive_treasury_tipped_occupation_code(
        np.array([4040, 4110, 4230, 2770, -1, 9999, np.nan])
    )
    np.testing.assert_array_equal(derived, [101, 102, 304, 208, 0, 0, 0])


# --------------------------------------------------------------------------- #
# QRF imputation
# --------------------------------------------------------------------------- #
def test_impute_is_deterministic_for_a_seed() -> None:
    person = _person_rows(90)
    donor = _donor_table()
    first = impute_us_sipp_tips(person, donor, seed=7, n_estimators=8)
    second = impute_us_sipp_tips(person, donor, seed=7, n_estimators=8)
    pd.testing.assert_frame_equal(first, second)


def test_impute_caps_large_donor_with_retired_named_seed(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}

    class FakeFittedQRF:
        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"tip_income": np.zeros(len(features), dtype=np.float64)},
                index=features.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs) -> None:
            pass

        def fit(self, frame: pd.DataFrame, **kwargs) -> FakeFittedQRF:
            captured["row_ids"] = frame["employment_income"].to_numpy()
            return FakeFittedQRF()

    monkeypatch.setattr("populace.fit.QRF", FakeQRF)
    donor = _donor_table(10_050)
    donor["employment_income"] = np.arange(len(donor), dtype=np.float64)

    impute_us_sipp_tips(_person_rows(20), donor, seed=7, n_estimators=3)

    expected_positions = np.sort(
        np.random.default_rng(5_559_651_045_748_063_828).choice(
            len(donor), size=10_000, replace=False
        )
    )
    np.testing.assert_array_equal(captured["row_ids"], expected_positions)


def test_impute_carries_code_without_erasing_tips_outside_list(monkeypatch) -> None:
    class FakeFittedQRF:
        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"tip_income": np.arange(len(features), dtype=np.float64) + 100.0},
                index=features.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs) -> None:
            pass

        def fit(self, *args, **kwargs) -> FakeFittedQRF:
            return FakeFittedQRF()

    monkeypatch.setattr("populace.fit.QRF", FakeQRF)
    person = _person_rows(90)
    result = impute_us_sipp_tips(
        person,
        _donor_table(),
        seed=42,
        n_estimators=8,
    )
    assert tuple(result.columns) == SIPP_TIP_OUTPUT_COLUMNS
    expected_codes = derive_treasury_tipped_occupation_code(person["PEIOOCC"])
    np.testing.assert_array_equal(
        result["treasury_tipped_occupation_code"], expected_codes
    )
    tips = result["tip_income"].to_numpy()
    assert (tips >= 0).all()
    assert np.all(tips[expected_codes == 0] > 0.0)
    assert tips[expected_codes > 0].sum() > 0.0


def test_impute_missing_donor_column_raises() -> None:
    with pytest.raises(ValueError, match="donor table missing column"):
        impute_us_sipp_tips(
            _person_rows(20),
            _donor_table().drop(columns=["tip_income"]),
            seed=0,
            n_estimators=5,
        )


# --------------------------------------------------------------------------- #
# Frame integration and healing
# --------------------------------------------------------------------------- #
def test_with_inputs_heals_default_surface_then_is_idempotent() -> None:
    person = _person_rows(90)
    person["tip_income"] = 0.0
    person["treasury_tipped_occupation_code"] = 0
    frame = _us_frame(person)
    donor = _donor_table()

    healed = with_us_sipp_tip_inputs(
        frame,
        seed=42,
        time_period=TIME_PERIOD,
        sipp_donor=donor,
    )
    healed_person = healed.table("person")
    assert healed_person["tip_income"].sum() > 0.0
    assert (healed_person["treasury_tipped_occupation_code"] > 0).any()

    repeated = with_us_sipp_tip_inputs(
        healed,
        seed=99,
        time_period=TIME_PERIOD,
        sipp_donor=donor,
    )
    for column in SIPP_TIP_OUTPUT_COLUMNS:
        np.testing.assert_array_equal(
            healed_person[column].to_numpy(),
            repeated.table("person")[column].to_numpy(),
        )


# --------------------------------------------------------------------------- #
# Gate and summary
# --------------------------------------------------------------------------- #
def test_signal_gate_passes_plausible_surface() -> None:
    gate = us_sipp_tips_signal_gate(_frame_with_tip_surface())
    assert gate.passed, gate.failures


def test_signal_gate_fails_when_columns_missing() -> None:
    gate = us_sipp_tips_signal_gate(_us_frame(_person_rows(20)))
    assert not gate.passed
    assert any("missing" in failure for failure in gate.failures)


def test_signal_gate_fails_on_constant_default_surface() -> None:
    frame = _frame_with_tip_surface(tipped_count=0, positive_tip_count=0)
    gate = us_sipp_tips_signal_gate(frame)
    assert not gate.passed
    assert any("constant" in failure for failure in gate.failures)


def test_signal_gate_fails_implausible_nonzero_shares() -> None:
    frame = _frame_with_tip_surface(
        n=100,
        tipped_count=50,
        positive_tip_count=40,
    )
    gate = us_sipp_tips_signal_gate(frame)
    assert not gate.passed
    assert any("tip-income share" in failure for failure in gate.failures)
    assert any("tipped-occupation share" in failure for failure in gate.failures)


def test_summary_reports_weighted_shares_bands_and_unique_counts() -> None:
    summary = us_sipp_tips_summary(_frame_with_tip_surface())
    assert summary["tip_income_nonzero_share"] == pytest.approx(0.008)
    assert summary["tipped_occupation_share"] == pytest.approx(0.07)
    assert summary["tip_income_nonzero_share_band"] == [0.001, 0.03]
    assert summary["tipped_occupation_share_band"] == [0.02, 0.15]
    assert summary["unique_counts"]["tip_income"] > 1
    assert summary["unique_counts"]["treasury_tipped_occupation_code"] > 1


# --------------------------------------------------------------------------- #
# SHA-verified provisioning cache
# --------------------------------------------------------------------------- #
def test_fetch_reuses_cache_only_when_sha_matches(tmp_path, monkeypatch) -> None:
    payload = b"synthetic pinned SIPP donor"
    cached = tmp_path / "pu2023_slim.csv"
    cached.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    def unexpected_network_call(*args, **kwargs):
        raise AssertionError("matching cache entry must not hit the network")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_network_call)
    result = fetch_sipp_2023_tip_donor(
        cache_dir=tmp_path,
        expected_sha256=digest,
    )
    assert result == cached
    assert result.read_bytes() == payload
