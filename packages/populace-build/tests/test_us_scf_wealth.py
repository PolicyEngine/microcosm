"""US SCF financial-asset (SSI countable-resource) stage tests (populace #356/#368).

The three asset leaves ``bank_account_assets`` / ``stock_assets`` /
``bond_assets`` are what ``ssi_countable_resources`` sums; with them absent the
SSI resource-limit reform class scores $0 (the #356 failure). This stage
SCF-imputes them, head-carried onto the household reference person.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceStageSpec
from populace.build.us_runtime import (
    SCF_FINANCIAL_ASSET_TARGET_COMPONENTS,
    SCF_WEALTH_PREDICTORS,
    US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
    US_SCF_WEALTH_STAGE_NAME,
    fetch_scf_2022_summary_extract,
    impute_us_scf_financial_assets,
    load_scf_2022_financial_asset_donor,
    us_scf_wealth_signal_gate,
    us_scf_wealth_stage_spec,
    us_scf_wealth_summary,
    with_us_scf_wealth_inputs,
)
from populace.build.us_runtime.scf_wealth import (
    _household_head_mask,
    _recipient_cps_race,
    _replace_sentinels,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024

_DONOR_WEIGHT_COLUMN = "scf_weight"


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
def _raw_scf_summary() -> pd.DataFrame:
    """A tiny SCF-summary-extract-shaped table (the columns the loader reads)."""

    rng = np.random.default_rng(0)
    n = 400
    liq = rng.gamma(2.0, 3_000.0, n)
    return pd.DataFrame(
        {
            "liq": liq,
            "stocks": rng.gamma(1.0, 5_000.0, n),
            "nmmf": rng.gamma(1.0, 4_000.0, n),
            "bond": np.where(rng.random(n) < 0.05, rng.gamma(1.0, 9_000.0, n), 0.0),
            "wgt": rng.uniform(500.0, 2_000.0, n),
            "age": rng.integers(20, 85, n).astype(float),
            "hhsex": rng.integers(1, 3, n).astype(float),
            "racecl5": rng.integers(1, 6, n).astype(float),
            "married": (rng.random(n) < 0.5).astype(float),
            "kids": rng.integers(0, 4, n).astype(float),
            "wageinc": rng.gamma(2.0, 20_000.0, n),
            "intdivinc": rng.gamma(1.0, 1_000.0, n),
            "ssretinc": rng.gamma(1.0, 8_000.0, n),
        }
    )


def _donor_table() -> pd.DataFrame:
    """A ready-made donor table (as the loader would emit) for impute tests."""

    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({p: rng.normal(0.0, 1.0, n) for p in SCF_WEALTH_PREDICTORS})
    frame["age"] = rng.integers(18, 90, n).astype(float)
    frame["is_female"] = (rng.random(n) < 0.5).astype(float)
    frame["cps_race"] = rng.integers(1, 8, n).astype(float)
    frame["is_married"] = (rng.random(n) < 0.5).astype(float)
    frame["own_children_in_household"] = rng.integers(0, 4, n).astype(float)
    frame["employment_income"] = rng.gamma(2.0, 15_000.0, n)
    frame["interest_dividend_income"] = rng.gamma(1.0, 900.0, n)
    frame["social_security_pension_income"] = rng.gamma(1.0, 7_000.0, n)
    frame["bank_account_assets"] = rng.gamma(2.0, 3_000.0, n)
    frame["stock_assets"] = np.where(
        rng.random(n) < 0.25, rng.gamma(1.0, 20_000.0, n), 0.0
    )
    frame["bond_assets"] = np.where(
        rng.random(n) < 0.04, rng.gamma(1.0, 9_000.0, n), 0.0
    )
    frame[_DONOR_WEIGHT_COLUMN] = rng.uniform(500.0, 2_000.0, n)
    return frame


def _person_rows(n_households: int = 60) -> pd.DataFrame:
    """A raw-ASEC-shaped recipient person table: two persons per household."""

    records: list[dict] = []
    rng = np.random.default_rng(2)
    person_id = 1
    for household in range(1, n_households + 1):
        # Head (line 1) then a second member (line 2).
        records.append(
            {
                "person_id": person_id,
                "person_household_id": household,
                "PH_SEQ": household,
                "A_LINENO": 1,
                "age": float(rng.integers(30, 85)),
                "is_female": bool(rng.integers(0, 2)),
                "PRDTRACE": int(rng.integers(1, 7)),
                "PRDTHSP": int(rng.integers(0, 2)),
                "A_MARITL": int(rng.integers(1, 8)),
                "PEPAR1": -1,
                "PEPAR2": -1,
                "employment_income_before_lsr": float(rng.gamma(2.0, 12_000.0)),
                "taxable_interest_income": float(rng.gamma(1.0, 500.0)),
                "social_security_retirement": float(rng.gamma(1.0, 6_000.0)),
            }
        )
        person_id += 1
        records.append(
            {
                "person_id": person_id,
                "person_household_id": household,
                "PH_SEQ": household,
                "A_LINENO": 2,
                "age": float(rng.integers(1, 60)),
                "is_female": bool(rng.integers(0, 2)),
                "PRDTRACE": int(rng.integers(1, 7)),
                "PRDTHSP": int(rng.integers(0, 2)),
                "A_MARITL": int(rng.integers(1, 8)),
                "PEPAR1": -1,
                "PEPAR2": -1,
                "employment_income_before_lsr": float(rng.gamma(1.0, 3_000.0)),
                "taxable_interest_income": 0.0,
                "social_security_retirement": 0.0,
            }
        )
        person_id += 1
    return pd.DataFrame(records)


def _us_frame(person: pd.DataFrame, *, weights: list[float] | None = None) -> Frame:
    person = person.copy()
    n = len(person)
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": unique_households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unique_households + 2_000}),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype="int64") + 4_000}
        ),
    }
    w = weights or [1.0] * len(unique_households)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray(w, dtype=np.float64), kind=WeightKind.DESIGN
            )
        },
    )


# --------------------------------------------------------------------------- #
# Manifest declaration                                                          #
# --------------------------------------------------------------------------- #
def test_stage_spec_loads_and_declares_outputs() -> None:
    spec = us_scf_wealth_stage_spec()
    assert isinstance(spec, SourceStageSpec)
    assert spec.stage == US_SCF_WEALTH_STAGE_NAME
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        assert column in spec.outputs


def test_output_columns_are_the_ssi_countable_resource_leaves() -> None:
    assert US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS == (
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
    )


def test_stock_target_sums_stocks_and_nmmf() -> None:
    assert SCF_FINANCIAL_ASSET_TARGET_COMPONENTS["stock_assets"] == ("stocks", "nmmf")
    assert SCF_FINANCIAL_ASSET_TARGET_COMPONENTS["bank_account_assets"] == ("liq",)
    assert SCF_FINANCIAL_ASSET_TARGET_COMPONENTS["bond_assets"] == ("bond",)


# --------------------------------------------------------------------------- #
# Donor loading                                                                 #
# --------------------------------------------------------------------------- #
def test_load_donor_derives_targets_predictors_and_weight(tmp_path) -> None:
    raw = _raw_scf_summary()
    path = tmp_path / "rscfp2022.dta"
    raw.to_stata(path, write_index=False)
    donor = load_scf_2022_financial_asset_donor(path)
    for column in (
        *US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        *SCF_WEALTH_PREDICTORS,
        _DONOR_WEIGHT_COLUMN,
    ):
        assert column in donor.columns
    # stock_assets is the stocks + nmmf sum.
    expected_stock = raw["stocks"].to_numpy() + raw["nmmf"].to_numpy()
    np.testing.assert_allclose(
        np.sort(donor["stock_assets"].to_numpy()), np.sort(expected_stock), rtol=1e-6
    )
    # All weights positive; targets non-negative.
    assert (donor[_DONOR_WEIGHT_COLUMN] > 0).all()
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        assert (donor[column] >= 0).all()


def test_load_donor_missing_column_raises(tmp_path) -> None:
    raw = _raw_scf_summary().drop(columns=["liq"])
    path = tmp_path / "rscfp2022.dta"
    raw.to_stata(path, write_index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_scf_2022_financial_asset_donor(path)


def test_replace_sentinels_zeroes_scf_missing_codes() -> None:
    series = pd.Series([-1.0, -7.0, -8.0, -9.0, 5_000.0, 0.0])
    cleaned = _replace_sentinels(series).to_numpy()
    np.testing.assert_array_equal(cleaned, [0.0, 0.0, 0.0, 0.0, 5_000.0, 0.0])


# --------------------------------------------------------------------------- #
# Predictor construction                                                        #
# --------------------------------------------------------------------------- #
def test_recipient_cps_race_mapping() -> None:
    person = pd.DataFrame(
        {
            "PRDTRACE": [1, 2, 4, 3, 1, 5],
            "PRDTHSP": [0, 0, 0, 0, 2, 0],
        }
    )
    race = _recipient_cps_race(person)
    # White, Black, Asian, Other(race 3), Hispanic-overrides, Other(race 5).
    np.testing.assert_array_equal(race, [1.0, 2.0, 4.0, 7.0, 3.0, 7.0])


def test_household_head_mask_is_one_per_household_at_lowest_line() -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [5, 5, 5, 9, 9],
            "A_LINENO": [3, 1, 2, 2, 1],
        }
    )
    mask = _household_head_mask(person)
    # Head of hh 5 is the A_LINENO==1 row (index 1); head of hh 9 is index 4.
    np.testing.assert_array_equal(mask, [False, True, False, False, True])
    assert mask.sum() == 2


# --------------------------------------------------------------------------- #
# Imputation (head-carry)                                                        #
# --------------------------------------------------------------------------- #
def test_impute_head_carries_and_zeroes_non_heads() -> None:
    person = _person_rows(60)
    donor = _donor_table()
    result = impute_us_scf_financial_assets(person, donor, seed=42, n_estimators=20)
    assert list(result.columns) == list(US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS)
    assert len(result) == len(person)
    head_mask = _household_head_mask(person)
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        values = result[column].to_numpy()
        assert (values >= 0).all()  # non-negative
        # Every non-head person carries exactly $0 (head-carry).
        assert np.all(values[~head_mask] == 0.0)
    # Heads carry signal: bank assets are not all zero on the heads.
    assert result["bank_account_assets"].to_numpy()[head_mask].sum() > 0


def test_impute_is_deterministic_for_a_seed() -> None:
    person = _person_rows(40)
    donor = _donor_table()
    a = impute_us_scf_financial_assets(person, donor, seed=7, n_estimators=15)
    b = impute_us_scf_financial_assets(person, donor, seed=7, n_estimators=15)
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        np.testing.assert_array_equal(a[column].to_numpy(), b[column].to_numpy())


def test_impute_missing_donor_column_raises() -> None:
    person = _person_rows(10)
    donor = _donor_table().drop(columns=["bond_assets"])
    with pytest.raises(ValueError, match="donor table missing column"):
        impute_us_scf_financial_assets(person, donor, seed=0, n_estimators=10)


# --------------------------------------------------------------------------- #
# Frame integration                                                             #
# --------------------------------------------------------------------------- #
def test_with_inputs_writes_all_three_columns() -> None:
    frame = _us_frame(_person_rows(60))
    donor = _donor_table()
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    person = out.table("person")
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        assert column in person.columns
        assert person[column].to_numpy().dtype == np.float64
    assert person["bank_account_assets"].to_numpy().sum() > 0


def test_with_inputs_is_idempotent_when_signal_present() -> None:
    frame = _us_frame(_person_rows(60))
    donor = _donor_table()
    once = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    twice = with_us_scf_wealth_inputs(
        once, seed=99, time_period=TIME_PERIOD, scf_donor=donor
    )
    # Passing through untouched: the second call (different seed) does not
    # re-impute because the surface already carries signal.
    np.testing.assert_array_equal(
        once.table("person")["bank_account_assets"].to_numpy(),
        twice.table("person")["bank_account_assets"].to_numpy(),
    )


def test_with_inputs_reimputes_when_bank_assets_constant() -> None:
    person = _person_rows(60)
    person["bank_account_assets"] = 0.0  # the engine-default landmine
    person["stock_assets"] = 0.0
    person["bond_assets"] = 0.0
    frame = _us_frame(person)
    donor = _donor_table()
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    assert out.table("person")["bank_account_assets"].to_numpy().sum() > 0


# --------------------------------------------------------------------------- #
# Gate + summary                                                                #
# --------------------------------------------------------------------------- #
def test_signal_gate_passes_on_imputed_surface() -> None:
    frame = _us_frame(_person_rows(200))
    donor = _donor_table()
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=donor
    )
    gate = us_scf_wealth_signal_gate(out)
    assert gate.passed, gate.failures


def test_signal_gate_fails_when_columns_missing() -> None:
    frame = _us_frame(_person_rows(4))
    gate = us_scf_wealth_signal_gate(frame)
    assert not gate.passed
    assert any("missing" in f for f in gate.failures)


def test_signal_gate_fails_on_constant_zero_surface() -> None:
    person = _person_rows(20)
    person["bank_account_assets"] = 0.0
    person["stock_assets"] = 0.0
    person["bond_assets"] = 0.0
    frame = _us_frame(person)
    gate = us_scf_wealth_signal_gate(frame)
    assert not gate.passed
    assert any("constant" in f or "nonzero share" in f for f in gate.failures)


def test_summary_reports_shares_and_bands() -> None:
    frame = _us_frame(_person_rows(200))
    out = with_us_scf_wealth_inputs(
        frame, seed=42, time_period=TIME_PERIOD, scf_donor=_donor_table()
    )
    summary = us_scf_wealth_summary(out)
    assert 0.0 <= summary["bank_account_assets_nonzero_share"] <= 1.0
    assert "bank_nonzero_share_band" in summary
    assert summary["unique_counts"]["bank_account_assets"] >= 2


# --------------------------------------------------------------------------- #
# Provisioning helper                                                           #
# --------------------------------------------------------------------------- #
def test_fetch_returns_cached_file_without_network(tmp_path) -> None:
    cached = tmp_path / "rscfp2022.dta"
    cached.write_bytes(b"stub")
    # A pre-existing non-empty cache file is returned as-is (no network call).
    result = fetch_scf_2022_summary_extract(cache_dir=tmp_path)
    assert result == cached
    assert result.read_bytes() == b"stub"
