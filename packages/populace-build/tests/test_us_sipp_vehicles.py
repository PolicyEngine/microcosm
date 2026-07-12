"""Tests for the SHA-pinned SIPP household-vehicle source stage."""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from populace.build.us_runtime.sipp_vehicles import (
    _OWNED_OBSERVED_COLUMN,
    _VALUE_OBSERVED_COLUMN,
    ARCHIVED_SIPP_VEHICLE_IMPUTE_URL,
    ARCHIVED_SIPP_VEHICLE_RECEIVER_URL,
    ARCHIVED_SIPP_VEHICLE_SOURCE_URL,
    ARCHIVED_SIPP_VEHICLE_TRANSFORM_URL,
    SIPP_2023_VEHICLE_DONOR_REVISION,
    SIPP_2023_VEHICLE_DONOR_SHA256,
    SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    SIPP_2023_VEHICLE_DONOR_URL,
    SIPP_VEHICLE_MODEL_PREDICTORS,
    SIPP_VEHICLE_SOURCE_COLUMNS,
    US_SIPP_VEHICLE_NONCONSTANT_HOUSEHOLD_COLUMNS,
    US_SIPP_VEHICLE_OUTPUT_COLUMNS,
    _append_owned_dummies,
    _predictor_encoding,
    _recipient_household_predictor_table,
    fetch_sipp_2023_vehicle_donor,
    impute_us_sipp_vehicles,
    load_sipp_2023_vehicle_donor,
    us_sipp_vehicles_signal_gate,
    us_sipp_vehicles_stage_spec,
    us_sipp_vehicles_summary,
    with_us_sipp_vehicle_inputs,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2026


def _source_row(
    household_id: int,
    person_number: int,
    *,
    month: int = 12,
    weight: float = 100.0,
    age: float = 40.0,
    sex: int = 1,
    marital_status: int = 1,
    total_income: float = 1_000.0,
    bank_income: float = 10.0,
    stock_income: float = 5.0,
    bond_income: float = 2.0,
    rental_income: float = 3.0,
    vehicles_owned: float = 2.0,
    vehicle_value: float = 30_000.0,
    home_value: float = 100_000.0,
    owned_status: int = 1,
    household_value_status: int = 1,
    vehicle_1_status: int = 1,
    vehicle_2_status: int = 1,
    vehicle_3_status: int = 1,
) -> dict[str, float | int]:
    return {
        "SSUID": household_id,
        "PNUM": person_number,
        "MONTHCODE": month,
        "WPFINWGT": weight,
        "TAGE": age,
        "ESEX": sex,
        "EMS": marital_status,
        "TPTOTINC": total_income,
        "TINC_BANK": bank_income,
        "TINC_STMF": stock_income,
        "TINC_BOND": bond_income,
        "TINC_RENT": rental_income,
        "TVEH_NUM": vehicles_owned,
        "THVAL_VEH": vehicle_value,
        "THVAL_HOME": home_value,
        "AVEH_NUM": owned_status,
        "AHVAL_VEH": household_value_status,
        "AVEH1VAL": vehicle_1_status,
        "AVEH2VAL": vehicle_2_status,
        "AVEH3VAL": vehicle_3_status,
    }


def _write_sipp_source(tmp_path: Path) -> Path:
    rows = [
        # A November record with extreme values must not enter the household.
        _source_row(
            10,
            1,
            month=11,
            total_income=999_999.0,
            vehicles_owned=5,
            vehicle_value=185_150.0,
        ),
        _source_row(10, 1, household_value_status=5),
        _source_row(
            10,
            2,
            age=10,
            sex=2,
            marital_status=2,
            total_income=100.0,
            bank_income=1.0,
            stock_income=0.0,
            bond_income=0.0,
            rental_income=0.0,
            household_value_status=5,
        ),
        _source_row(
            20,
            1,
            age=30,
            sex=2,
            marital_status=2,
            vehicles_owned=0,
            vehicle_value=0,
            home_value=0,
            owned_status=0,
            household_value_status=0,
            vehicle_1_status=0,
            vehicle_2_status=0,
            vehicle_3_status=0,
        ),
        # Count is allocated, but value is observed.
        _source_row(
            30,
            1,
            age=70,
            vehicles_owned=3,
            vehicle_value=45_000,
            owned_status=2,
        ),
        # Count is observed, but a component value is allocated with status 5.
        _source_row(
            40,
            1,
            vehicles_owned=1,
            vehicle_value=9_000,
            vehicle_1_status=5,
        ),
        # Invalid survey weight: removed even though both targets are observed.
        _source_row(50, 1, weight=0.0),
    ]
    source = pd.DataFrame(rows, columns=SIPP_VEHICLE_SOURCE_COLUMNS)
    path = tmp_path / "pu2023.csv"
    source.to_csv(path, sep="|", index=False)
    return path


def _ready_donor(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(71)
    income = rng.gamma(2.0, 22_000.0, n)
    homeowner = rng.integers(0, 2, n).astype(np.float64)
    count = np.clip((income // 28_000).astype(int) + homeowner.astype(int), 0, 5)
    # Retain a sizeable structural-zero class for both output gates.
    count[income < 18_000] = 0
    value = np.where(
        count > 0,
        count * 7_500.0 + rng.gamma(1.5, 2_000.0, n),
        0.0,
    )
    donor = pd.DataFrame(
        {
            "household_employment_income": income,
            "household_interest_income": rng.gamma(1.0, 500.0, n),
            "household_dividend_income": rng.gamma(1.0, 400.0, n),
            "household_rental_income": np.where(
                rng.random(n) < 0.12, rng.gamma(1.0, 3_000.0, n), 0.0
            ),
            "reference_age": rng.integers(19, 90, n).astype(np.float64),
            "reference_is_female": rng.integers(0, 2, n).astype(np.float64),
            "reference_is_married": rng.integers(0, 2, n).astype(np.float64),
            "count_under_18": rng.integers(0, 5, n).astype(np.float64),
            "household_size": rng.integers(1, 7, n).astype(np.float64),
            "is_homeowner": homeowner,
            "household_vehicles_owned": count.astype(np.float64),
            "household_vehicles_value": value,
            "household_weight": rng.uniform(500.0, 2_500.0, n),
            _OWNED_OBSERVED_COLUMN: True,
            _VALUE_OBSERVED_COLUMN: True,
        }
    )
    return donor


def _recipient_frame(n_households: int = 90) -> Frame:
    rng = np.random.default_rng(72)
    people: list[dict[str, object]] = []
    person_id = 1
    for household_id in range(1, n_households + 1):
        homeowner = household_id % 3 != 0
        head_income = float((household_id % 9) * 9_000 + 3_000)
        # Put line 2 first to prove that A_LINENO, not input order, selects head.
        people.append(
            {
                "person_id": person_id,
                "person_household_id": household_id,
                "person_spm_unit_id": household_id + 2_000,
                "A_LINENO": 2,
                "age": float(rng.integers(2, 17)),
                "is_female": bool(rng.integers(0, 2)),
                "A_MARITL": 7,
                "employment_income_before_lsr": 0.0,
                "taxable_interest_income": 0.0,
                "tax_exempt_interest_income": 0.0,
                "qualified_dividend_income": 0.0,
                "non_qualified_dividend_income": 0.0,
                "rental_income": 0.0,
                "SPM_TENMORTSTATUS": 1 if homeowner else 3,
            }
        )
        person_id += 1
        people.append(
            {
                "person_id": person_id,
                "person_household_id": household_id,
                "person_spm_unit_id": household_id + 2_000,
                "A_LINENO": 1,
                "age": float(25 + household_id % 50),
                "is_female": bool(household_id % 2),
                "A_MARITL": 1 if household_id % 2 else 5,
                "employment_income_before_lsr": head_income,
                "taxable_interest_income": float(household_id % 5 * 80),
                "tax_exempt_interest_income": float(household_id % 5 * 20),
                "qualified_dividend_income": float(household_id % 4 * 50),
                "non_qualified_dividend_income": float(household_id % 4 * 25),
                "rental_income": float(household_id % 7 == 0) * 2_000.0,
                "SPM_TENMORTSTATUS": 1 if homeowner else 3,
            }
        )
        person_id += 1
    person = pd.DataFrame(people)
    household_ids = np.arange(1, n_households + 1, dtype=np.int64)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    # Duplicate marital-unit ids exercise the archived pairing fallback.
    person["person_marital_unit_id"] = person["person_household_id"] + 4_000
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "net_worth": np.linspace(50_000.0, 150_000.0, n_households),
        }
    )
    tables = {
        "person": person,
        "household": household,
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": np.arange(1, n_households + 1) + 1_000}
        ),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.arange(1, n_households + 1) + 2_000}
        ),
        "family": pd.DataFrame(
            {"family_id": np.arange(1, n_households + 1) + 3_000}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(1, n_households + 1) + 4_000}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.full(n_households, 1_000_000.0), WeightKind.DESIGN
            )
        },
    )


def _replace_household(frame: Frame, **columns: np.ndarray) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in columns.items():
        tables["household"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def test_immutable_artifact_and_archive_coordinates_are_exact() -> None:
    assert SIPP_2023_VEHICLE_DONOR_REVISION == (
        "21280dca5995e978d706740a8a4b9b7860cfd7b6"
    )
    assert SIPP_2023_VEHICLE_DONOR_SHA256 == (
        "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
    )
    assert SIPP_2023_VEHICLE_DONOR_SIZE_BYTES == 3_726_010_471
    assert SIPP_2023_VEHICLE_DONOR_REVISION in SIPP_2023_VEHICLE_DONOR_URL
    assert SIPP_2023_VEHICLE_DONOR_URL.endswith("/pu2023.csv")
    assert "/resolve/" in SIPP_2023_VEHICLE_DONOR_URL
    for url in (
        ARCHIVED_SIPP_VEHICLE_SOURCE_URL,
        ARCHIVED_SIPP_VEHICLE_TRANSFORM_URL,
        ARCHIVED_SIPP_VEHICLE_IMPUTE_URL,
        ARCHIVED_SIPP_VEHICLE_RECEIVER_URL,
    ):
        assert "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe" in url
        retired_repository = "policyengine-" + "us-data"
        assert f"github.com/PolicyEngine/{retired_repository}/blob/" in url


def test_exact_source_columns_predictors_outputs_and_stage_spec() -> None:
    assert SIPP_VEHICLE_SOURCE_COLUMNS == (
        "SSUID",
        "PNUM",
        "MONTHCODE",
        "WPFINWGT",
        "TAGE",
        "ESEX",
        "EMS",
        "TPTOTINC",
        "TINC_BANK",
        "TINC_STMF",
        "TINC_BOND",
        "TINC_RENT",
        "TVEH_NUM",
        "THVAL_VEH",
        "THVAL_HOME",
        "AVEH_NUM",
        "AHVAL_VEH",
        "AVEH1VAL",
        "AVEH2VAL",
        "AVEH3VAL",
    )
    assert SIPP_VEHICLE_MODEL_PREDICTORS == (
        "household_employment_income",
        "household_interest_income",
        "household_dividend_income",
        "household_rental_income",
        "reference_age",
        "reference_is_female",
        "reference_is_married",
        "count_under_18",
        "household_size",
        "is_homeowner",
    )
    assert US_SIPP_VEHICLE_OUTPUT_COLUMNS == (
        "household_vehicles_owned",
        "household_vehicles_value",
    )
    assert US_SIPP_VEHICLE_NONCONSTANT_HOUSEHOLD_COLUMNS == (
        US_SIPP_VEHICLE_OUTPUT_COLUMNS
    )
    spec = us_sipp_vehicles_stage_spec()
    assert spec.stage == "vehicle_assets"
    assert set(US_SIPP_VEHICLE_OUTPUT_COLUMNS) <= set(spec.outputs)


def test_loader_applies_exact_december_household_transform_and_masks(tmp_path) -> None:
    donor = load_sipp_2023_vehicle_donor(
        _write_sipp_source(tmp_path), expected_size_bytes=None, chunksize=2
    ).set_index("household_id")

    assert donor.index.tolist() == [10, 20, 30, 40]
    household_10 = donor.loc[10]
    assert household_10["household_weight"] == pytest.approx(100.0)
    assert household_10["household_employment_income"] == pytest.approx(13_200.0)
    assert household_10["household_interest_income"] == pytest.approx(156.0)
    assert household_10["household_dividend_income"] == pytest.approx(60.0)
    assert household_10["household_rental_income"] == pytest.approx(36.0)
    assert household_10["reference_age"] == pytest.approx(40.0)
    assert household_10["reference_is_female"] == pytest.approx(0.0)
    assert household_10["reference_is_married"] == pytest.approx(1.0)
    assert household_10["count_under_18"] == pytest.approx(1.0)
    assert household_10["household_size"] == pytest.approx(2.0)
    assert household_10["is_homeowner"] == pytest.approx(1.0)
    assert household_10["household_vehicles_owned"] == pytest.approx(2.0)
    assert household_10["household_vehicles_value"] == pytest.approx(30_000.0)
    # AHVAL_VEH=5 does not invalidate value; archived masking uses AVEH1-3.
    assert bool(household_10[_OWNED_OBSERVED_COLUMN])
    assert bool(household_10[_VALUE_OBSERVED_COLUMN])

    assert not bool(donor.loc[30, _OWNED_OBSERVED_COLUMN])
    assert bool(donor.loc[30, _VALUE_OBSERVED_COLUMN])
    assert bool(donor.loc[40, _OWNED_OBSERVED_COLUMN])
    assert not bool(donor.loc[40, _VALUE_OBSERVED_COLUMN])
    # The November extreme must not leak into either target or predictor.
    assert donor.loc[10, "household_vehicles_owned"] != 5
    assert donor.loc[10, "household_employment_income"] < 100_000


def test_loader_rejects_missing_columns_and_bad_hash(tmp_path) -> None:
    path = _write_sipp_source(tmp_path)
    with pytest.raises(ValueError, match="sha-256 verification"):
        load_sipp_2023_vehicle_donor(
            path,
            expected_sha256="0" * 64,
            expected_size_bytes=None,
        )

    missing_path = tmp_path / "missing.csv"
    pd.DataFrame({"SSUID": [1], "MONTHCODE": [12]}).to_csv(
        missing_path, sep="|", index=False
    )
    with pytest.raises(ValueError, match="missing column"):
        load_sipp_2023_vehicle_donor(missing_path, expected_size_bytes=None)


def test_cached_full_donor_matches_pinned_household_support() -> None:
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--policyengine--policyengine-" + "us-data")
        / "snapshots"
        / SIPP_2023_VEHICLE_DONOR_REVISION
        / "pu2023.csv"
    )
    if not snapshot.is_file():
        pytest.skip("the 3.73 GB pinned SIPP donor is not mounted")

    donor = load_sipp_2023_vehicle_donor(
        snapshot,
        expected_sha256=SIPP_2023_VEHICLE_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    )
    owned_observed = donor[_OWNED_OBSERVED_COLUMN].astype(bool)
    value_observed = donor[_VALUE_OBSERVED_COLUMN].astype(bool)
    assert len(donor) == 16_841
    assert int(owned_observed.sum()) == 16_840
    assert int(value_observed.sum()) == 8_829
    assert int((owned_observed & value_observed).sum()) == 8_828
    assert set(donor.loc[owned_observed, "household_vehicles_owned"].unique()) == {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    }
    assert int(
        (
            (donor["household_vehicles_value"] > 0)
            & (donor["household_vehicles_owned"] == 0)
        ).sum()
    ) == 87


class _ChunkedResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def test_fetch_streams_verifies_atomically_and_reuses_cache(tmp_path, monkeypatch) -> None:
    payload = b"small synthetic pinned donor payload"
    digest = hashlib.sha256(payload).hexdigest()
    response = _ChunkedResponse(payload)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    path = fetch_sipp_2023_vehicle_donor(
        tmp_path,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
        chunk_size=5,
    )
    assert path.read_bytes() == payload
    assert len(response.read_sizes) > 2
    assert all(size == 5 for size in response.read_sizes)
    assert not (tmp_path / "pu2023.csv.part").exists()

    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("valid cached donor should not be downloaded again")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_download)
    assert (
        fetch_sipp_2023_vehicle_donor(
            tmp_path,
            expected_sha256=digest,
            expected_size_bytes=len(payload),
            chunk_size=5,
        )
        == path
    )


def test_fetch_failure_removes_partial_without_replacing_existing(tmp_path, monkeypatch) -> None:
    target = tmp_path / "pu2023.csv"
    target.write_bytes(b"existing invalid cache")
    response = _ChunkedResponse(b"new but wrong payload")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(ValueError, match="sha-256 verification"):
        fetch_sipp_2023_vehicle_donor(
            tmp_path,
            expected_sha256="f" * 64,
            expected_size_bytes=len(b"new but wrong payload"),
            chunk_size=4,
        )
    assert target.read_bytes() == b"existing invalid cache"
    assert not (tmp_path / "pu2023.csv.part").exists()


def test_receiver_uses_line_one_head_household_sums_and_raw_tenure() -> None:
    frame = _recipient_frame(6)
    receiver = _recipient_household_predictor_table(frame)
    assert receiver.index.tolist() == [1, 2, 3, 4, 5, 6]
    # Head is line 1 even though line 2 appears first in the person table.
    assert receiver.loc[1, "reference_age"] == pytest.approx(26.0)
    assert receiver.loc[1, "reference_is_female"] == pytest.approx(1.0)
    assert receiver.loc[1, "reference_is_married"] == pytest.approx(1.0)
    assert receiver.loc[1, "household_employment_income"] == pytest.approx(12_000.0)
    assert receiver.loc[1, "count_under_18"] == pytest.approx(1.0)
    assert receiver.loc[1, "household_size"] == pytest.approx(2.0)
    assert receiver.loc[1, "is_homeowner"] == pytest.approx(1.0)
    assert receiver.loc[3, "is_homeowner"] == pytest.approx(0.0)


def test_count_is_weighted_deterministic_classifier_and_value_chains_on_dummies(
    monkeypatch,
) -> None:
    frame = _recipient_frame(45)
    donor = _ready_donor()
    calls: list[dict[str, object]] = []
    real_fit = RandomForestClassifier.fit

    def recording_fit(self, x, y, sample_weight=None):
        calls.append(
            {
                "model": self,
                "columns": tuple(x.columns),
                "sample_weight": np.asarray(sample_weight),
                "classes": tuple(sorted(pd.unique(y))),
            }
        )
        return real_fit(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(RandomForestClassifier, "fit", recording_fit)
    first = impute_us_sipp_vehicles(frame, donor, seed=42, n_estimators=12)
    second = impute_us_sipp_vehicles(frame, donor, seed=42, n_estimators=12)

    assert calls
    assert all(isinstance(call["model"], RandomForestClassifier) for call in calls)
    assert all(call["model"].random_state == 42 for call in calls)
    assert all(call["model"].n_estimators == 12 for call in calls)
    assert all(np.all(call["sample_weight"] > 0) for call in calls)
    assert calls[0]["classes"] == tuple(sorted(donor.household_vehicles_owned.unique()))
    pd.testing.assert_frame_equal(first, second)
    assert np.issubdtype(first["household_vehicles_owned"].dtype, np.integer)
    assert np.issubdtype(first["household_vehicles_value"].dtype, np.floating)
    assert (first["household_vehicles_owned"] >= 0).all()
    assert (first["household_vehicles_value"] >= 0).all()

    encoding = _predictor_encoding(donor)
    encoded = encoding.transform(donor)
    _, owned_dummy_columns = _append_owned_dummies(
        encoded,
        donor["household_vehicles_owned"],
        levels=tuple(sorted(donor["household_vehicles_owned"].unique())),
    )
    assert owned_dummy_columns
    assert all(name.startswith("household_vehicles_owned__") for name in owned_dummy_columns)


def test_frame_application_is_household_grain_idempotent_and_preserves_net_worth() -> None:
    frame = _recipient_frame()
    original_net_worth = frame.table("household")["net_worth"].copy()
    restored = with_us_sipp_vehicle_inputs(
        frame,
        seed=42,
        time_period=TIME_PERIOD,
        sipp_donor=_ready_donor(),
        n_estimators=16,
    )
    household = restored.table("household")
    assert set(US_SIPP_VEHICLE_OUTPUT_COLUMNS) <= set(household.columns)
    pd.testing.assert_series_equal(household["net_worth"], original_net_worth)
    assert len(household["household_vehicles_owned"]) == len(
        frame.table("household")
    )
    assert restored.weights_for("household") == frame.weights_for("household")

    passed_through = with_us_sipp_vehicle_inputs(
        restored,
        seed=999,
        time_period=TIME_PERIOD,
        sipp_donor=_ready_donor(40),
        n_estimators=4,
    )
    assert passed_through is restored


def test_summary_and_gate_require_broad_real_signal() -> None:
    frame = with_us_sipp_vehicle_inputs(
        _recipient_frame(),
        seed=42,
        time_period=TIME_PERIOD,
        sipp_donor=_ready_donor(),
        n_estimators=16,
    )
    summary = us_sipp_vehicles_summary(frame)
    gate = us_sipp_vehicles_signal_gate(frame)
    assert gate.passed, gate.failures
    assert summary["owned_noninteger_count"] == 0
    assert summary["weighted_totals"]["household_vehicles_owned"] > 0
    assert summary["weighted_totals"]["household_vehicles_value"] > 0
    assert summary["positive_value_with_positive_owned_weighted_share"] > 0


def test_gate_rejects_missing_constant_negative_and_fractional_surfaces() -> None:
    base = _recipient_frame(10)
    missing = us_sipp_vehicles_signal_gate(base)
    assert not missing.passed
    assert "missing" in missing.failures[0]

    constant = _replace_household(
        base,
        household_vehicles_owned=np.zeros(10),
        household_vehicles_value=np.zeros(10),
    )
    constant_gate = us_sipp_vehicles_signal_gate(constant)
    assert not constant_gate.passed
    assert any("constant" in failure for failure in constant_gate.failures)

    bad = _replace_household(
        base,
        household_vehicles_owned=np.array(
            [0.0, 1.5, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0]
        ),
        household_vehicles_value=np.array(
            [0.0, 5_000.0, -1.0, 8_000.0, 4_000.0, 7_000.0, 0.0, 3_000.0, 9_000.0, 2_000.0]
        ),
    )
    bad_gate = us_sipp_vehicles_signal_gate(bad)
    assert not bad_gate.passed
    assert any("non-integer" in failure for failure in bad_gate.failures)
    assert any("negative" in failure for failure in bad_gate.failures)
