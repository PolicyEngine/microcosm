"""SIPP financial-asset donor and imputer tests (microcosm #374)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.sipp_december_person_rows import (
    read_sipp_december_person_rows,
)
from microcosm.build.us_runtime.sipp_financial_assets import (
    SIPP_2023_FINANCIAL_ASSET_DONOR_DOWNLOAD_REVISION,
    SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS,
    SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE,
    SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    SIPP_2023_FINANCIAL_ASSET_DONOR_URL,
    SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN,
    SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS,
    SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
    SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS,
    SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS,
    _target_balanced_cap,
    build_sipp_2023_financial_asset_donor,
    fetch_sipp_2023_financial_asset_donor,
    impute_us_sipp_financial_assets,
)


def _raw_sipp() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    people = (
        ("001", 101, 42, 1, 1),
        ("001", 102, 4, 2, 6),
        ("002", 101, 67, 2, 1),
        ("002", 102, 17, 1, 6),
    )
    for month in (11, 12):
        for position, (ssuid, pnum, age, sex, marital_status) in enumerate(people):
            row = {column: 0.0 for column in SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS}
            row.update(
                {
                    "SSUID": ssuid,
                    "PNUM": pnum,
                    "MONTHCODE": month,
                    "WPFINWGT": 100.0 + position,
                    "TAGE": age,
                    "ESEX": sex,
                    "EMS": marital_status,
                    "TSSSAMT": 10.0 + position,
                    "TRETINCAMT": 20.0 + position,
                    "TVAL_BANK": 100.0 + position,
                    "TVAL_STMF": 200.0 + position,
                    "TVAL_BOND": 300.0 + position,
                    "TINC_BANK": 1.0 + position,
                    "TINC_STMF": 2.0 + position,
                    "TINC_BOND": 3.0 + position,
                    "TINC_RENT": 4.0 + position,
                }
            )
            for job_number in range(1, 8):
                row[f"TJB{job_number}_MSUM"] = float(job_number + position)
            for columns in SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS.values():
                for column in columns:
                    row[column] = 1.0
            rows.append(row)
    frame = pd.DataFrame(rows)
    december = frame["MONTHCODE"].eq(12)
    first_december = frame.index[december][0]
    frame.loc[first_december, "AJSSAVVAL"] = 2.0
    second_december = frame.index[december][1]
    frame.loc[second_december, "AJSSAVVAL"] = 9.0
    third_december = frame.index[december][2]
    frame.loc[third_december, "AJSSAVVAL"] = 0.0
    last_december = frame.index[december][-1]
    frame.loc[last_december, "TVAL_BOND"] = np.nan
    return frame


def _sipp_donor(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(10)
    donor = pd.DataFrame(
        {
            predictor: rng.normal(size=n)
            for predictor in SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS
        }
    )
    donor["age"] = rng.integers(18, 90, n)
    donor["is_female"] = rng.integers(0, 2, n)
    donor["is_married"] = rng.integers(0, 2, n)
    donor["count_under_18"] = rng.integers(0, 4, n)
    donor["count_under_6"] = rng.integers(0, 2, n)
    donor["household_size"] = rng.integers(1, 6, n)
    donor["employment_income"] = rng.gamma(1.5, 10_000.0, n)
    donor["social_security"] = rng.gamma(0.8, 4_000.0, n)
    donor["retirement_income"] = rng.gamma(0.8, 5_000.0, n)
    donor["non_ssi_income"] = (
        donor["employment_income"]
        + donor["social_security"]
        + donor["retirement_income"]
    )
    donor["bank_account_assets"] = np.where(
        rng.random(n) < 0.55, rng.gamma(1.0, 800.0, n), 0.0
    )
    donor["stock_assets"] = np.where(
        rng.random(n) < 0.10, rng.gamma(1.0, 2_000.0, n), 0.0
    )
    donor["bond_assets"] = np.where(
        rng.random(n) < 0.04, rng.gamma(1.0, 1_000.0, n), 0.0
    )
    donor[SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN] = rng.uniform(1.0, 4.0, n)
    for target in SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS:
        donor[f"{target}_is_observed"] = True
    return donor


def _recipient(n_households: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for household_id in range(1, n_households + 1):
        for line_number in (1, 2):
            rows.append(
                {
                    "person_household_id": household_id,
                    "A_LINENO": line_number,
                    "age": float(
                        rng.integers(25, 85)
                        if line_number == 1
                        else rng.integers(1, 60)
                    ),
                    "is_female": bool(rng.integers(0, 2)),
                    "is_married": line_number == 1 and household_id % 2 == 0,
                    "employment_income_before_lsr": float(rng.gamma(1.5, 8_000.0)),
                    "taxable_interest_income": float(rng.gamma(0.5, 200.0)),
                    "tax_exempt_interest_income": 0.0,
                    "qualified_dividend_income": float(rng.gamma(0.4, 100.0)),
                    "non_qualified_dividend_income": 0.0,
                    "rental_income": 0.0,
                    "social_security_retirement": float(rng.gamma(0.5, 2_000.0)),
                    "taxable_private_pension_income": float(rng.gamma(0.4, 1_500.0)),
                }
            )
    return pd.DataFrame(rows)


def test_pinned_sipp_source_and_final_archived_predictor_contract() -> None:
    assert SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION == (
        "21280dca5995e978d706740a8a4b9b7860cfd7b6"
    )
    assert SIPP_2023_FINANCIAL_ASSET_DONOR_DOWNLOAD_REVISION == "1.115.5"
    assert SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256 == (
        "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
    )
    assert SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES == 3_726_010_471
    assert "".join(SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS) == (
        "PolicyEngine/" + "policyengine-" + "us-data"
    )
    assert SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE == "model"
    assert SIPP_2023_FINANCIAL_ASSET_DONOR_URL.endswith(
        f"/resolve/{SIPP_2023_FINANCIAL_ASSET_DONOR_DOWNLOAD_REVISION}/pu2023.csv"
    )
    assert SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS == (
        "employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
        "social_security",
        "retirement_income",
        "non_ssi_income",
        "age",
        "is_female",
        "is_married",
        "count_under_18",
        "count_under_6",
        "household_size",
    )
    assert SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS == {
        "bank_account_assets": "TVAL_BANK",
        "stock_assets": "TVAL_STMF",
        "bond_assets": "TVAL_BOND",
    }


def test_donor_keeps_december_rows_and_matches_archived_transforms(tmp_path) -> None:
    raw = _raw_sipp()
    path = tmp_path / "pu2023.csv"
    raw.to_csv(path, sep="|", index=False)
    donor = build_sipp_2023_financial_asset_donor(
        read_sipp_december_person_rows(
            path,
            columns=SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
            chunksize=3,
        ),
        max_train_samples=None,
    )

    assert len(donor) == 4
    age_42 = donor.loc[donor["age"].eq(42)].iloc[0]
    assert age_42["employment_income"] == 12.0 * sum(range(1, 8))
    assert age_42["interest_income"] == 48.0
    assert age_42["dividend_income"] == 24.0
    assert age_42["rental_income"] == 48.0
    assert age_42["social_security"] == 120.0
    assert age_42["retirement_income"] == 240.0
    assert age_42["non_ssi_income"] == 696.0
    assert age_42["count_under_18"] == 1.0
    assert age_42["count_under_6"] == 1.0
    assert age_42["household_size"] == 2.0
    assert not bool(age_42["bank_account_assets_is_observed"])
    assert bool(age_42["stock_assets_is_observed"])
    assert bool(
        donor.loc[donor["age"].eq(4), "bank_account_assets_is_observed"].iloc[0]
    )
    assert bool(
        donor.loc[donor["age"].eq(67), "bank_account_assets_is_observed"].iloc[0]
    )
    age_17 = donor.loc[donor["age"].eq(17)].iloc[0]
    assert age_17["bond_assets"] == 0.0
    assert not bool(age_17["bond_assets_is_observed"])


def test_fetch_prefers_verified_local_file_without_hub_access(tmp_path) -> None:
    local = tmp_path / "pu2023.csv"
    local.write_bytes(b"tiny-fixture")
    resolved = fetch_sipp_2023_financial_asset_donor(
        local_path=local,
        expected_sha256=None,
        expected_size_bytes=len(b"tiny-fixture"),
    )
    assert resolved == local


def test_fetch_declares_pinned_hugging_face_contract(monkeypatch, tmp_path) -> None:
    import huggingface_hub

    downloaded = tmp_path / "downloaded-pu2023.csv"
    downloaded.write_bytes(b"remote-fixture")
    calls: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs):
        calls.update(kwargs)
        return str(downloaded)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    resolved = fetch_sipp_2023_financial_asset_donor(
        cache_dir=tmp_path / "cache",
        expected_sha256=None,
        expected_size_bytes=len(b"remote-fixture"),
    )

    assert resolved == downloaded
    assert calls["repo_id"] == "".join(
        SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS
    )
    assert calls["repo_type"] == SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE
    assert calls["revision"] == SIPP_2023_FINANCIAL_ASSET_DONOR_DOWNLOAD_REVISION
    assert calls["filename"] == "pu2023.csv"


def _plant_recorded_snapshot(cache_dir, payload: bytes):
    repository_id = "".join(SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS)
    snapshot = (
        cache_dir / f"{SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE}s--"
        f"{repository_id.replace('/', '--')}"
        / "snapshots"
        / SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION
        / "pu2023.csv"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(payload)
    return snapshot


def test_fetch_reuses_recorded_revision_snapshot_without_download(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub

    payload = b"recorded-snapshot-fixture"
    cache_dir = tmp_path / "cache"
    snapshot = _plant_recorded_snapshot(cache_dir, payload)

    def unexpected_download(**_kwargs):
        raise AssertionError("a verified recorded snapshot must not be downloaded")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)
    resolved = fetch_sipp_2023_financial_asset_donor(
        cache_dir=cache_dir,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )

    assert resolved == snapshot


def test_fetch_downloads_when_recorded_snapshot_fails_verification(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub

    payload = b"remote-fixture"
    cache_dir = tmp_path / "cache"
    _plant_recorded_snapshot(cache_dir, b"stale-fixture!")
    downloaded = tmp_path / "downloaded-pu2023.csv"
    downloaded.write_bytes(payload)
    calls: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs):
        calls.update(kwargs)
        return str(downloaded)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    resolved = fetch_sipp_2023_financial_asset_donor(
        cache_dir=cache_dir,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )

    assert resolved == downloaded
    assert calls["revision"] == SIPP_2023_FINANCIAL_ASSET_DONOR_DOWNLOAD_REVISION


def test_fetch_default_paths_reuse_recorded_snapshot_without_download(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub
    from huggingface_hub import constants

    payload = b"default-cache-fixture"
    home = tmp_path / "home"
    hub_cache = tmp_path / "hub"
    snapshot = _plant_recorded_snapshot(hub_cache, payload)

    def unexpected_download(**_kwargs):
        raise AssertionError("a verified recorded snapshot must not be downloaded")

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)
    resolved = fetch_sipp_2023_financial_asset_donor(
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )

    assert resolved == snapshot


def test_fetch_downloads_when_hub_cache_records_the_file_as_missing(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub

    payload = b"remote-fixture"
    cache_dir = tmp_path / "cache"
    snapshot = _plant_recorded_snapshot(cache_dir, payload)
    missing_marker = (
        snapshot.parents[2]
        / ".no_exist"
        / SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION
        / "pu2023.csv"
    )
    missing_marker.parent.mkdir(parents=True)
    missing_marker.touch()
    downloaded = tmp_path / "downloaded-pu2023.csv"
    downloaded.write_bytes(payload)

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **_kwargs: str(downloaded)
    )
    resolved = fetch_sipp_2023_financial_asset_donor(
        cache_dir=cache_dir,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )

    assert resolved == downloaded


def test_target_balanced_cap_pins_archived_sampling_seed() -> None:
    donor = pd.DataFrame(
        {
            "row": range(12),
            "bank_account_assets_is_observed": [i < 8 for i in range(12)],
            "stock_assets_is_observed": [2 <= i < 10 for i in range(12)],
            "bond_assets_is_observed": [4 <= i < 12 for i in range(12)],
        }
    )

    sampled = _target_balanced_cap(donor, max_train_samples=6)
    assert sampled["row"].tolist() == [0, 1, 4, 2, 5, 8]


def test_sipp_imputation_is_deterministic_and_head_carried() -> None:
    person = _recipient()
    donor = _sipp_donor()
    first = impute_us_sipp_financial_assets(
        person,
        donor,
        seed=23,
        n_estimators=12,
    )
    second = impute_us_sipp_financial_assets(
        person,
        donor,
        seed=23,
        n_estimators=12,
    )
    head = person["A_LINENO"].eq(1).to_numpy()
    for column in SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS:
        np.testing.assert_array_equal(first[column], second[column])
        assert (first.loc[~head, column] == 0.0).all()
    assert first.loc[head, "bank_account_assets"].sum() > 0.0
