"""SCF auto-loan and OBBBA qualifying-interest stage tests.

The retired eCPS imputed household auto-loan balance and annual interest from
the full 2022 SCF, conditioning on an SCF-style household reference person and
household-summed income.  The OBBBA deduction uses a later, distinct pure input
(``qualified_passenger_vehicle_loan_interest``), so this port also derives a
documented expected qualifying share rather than leaving the reform structural
zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import (
    QUALIFIED_AUTO_LOAN_ANNUAL_ISSUANCE_TARGET,
    SCF_2022_FULL_EXTRACT_MEMBER,
    SCF_2022_FULL_EXTRACT_MEMBER_SHA256,
    SCF_2022_FULL_EXTRACT_URL,
    SCF_2022_FULL_EXTRACT_ZIP_SHA256,
    SCF_AUTO_LOAN_AMOUNT_COLUMNS,
    SCF_AUTO_LOAN_RATE_COLUMNS,
    US_SCF_AUTO_LOAN_OUTPUT_COLUMNS,
    fetch_scf_2022_full_extract,
    impute_us_scf_auto_loans,
    load_scf_2022_auto_loan_donor,
    qualified_auto_loan_interest_proxy,
    us_scf_auto_loans_signal_gate,
    us_scf_auto_loans_stage_spec,
    us_scf_auto_loans_summary,
    with_us_scf_auto_loan_inputs,
)
from populace.build.us_runtime.scf_auto_loans import (
    _scf_reference_person_mask,
    _sha256_hexdigest,
)
from populace.build.us_runtime.scf_wealth import SCF_WEALTH_PREDICTORS
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024
_DONOR_WEIGHT_COLUMN = "scf_weight"


def _raw_scf_summary(n_families: int = 8) -> pd.DataFrame:
    """A summary-extract-shaped table with all five SCF implicates."""

    records: list[dict[str, float]] = []
    for family in range(1, n_families + 1):
        for implicate in range(1, 6):
            records.append(
                {
                    "y1": float(family),
                    "yy1": float(implicate),
                    "wgt": 1_000.0 + family,
                    "age": 30.0 + family,
                    "hhsex": 1.0 if family % 2 else 2.0,
                    "racecl5": float((family % 5) + 1),
                    "married": float(family % 2),
                    "kids": float(family % 3),
                    "wageinc": 20_000.0 + 1_000.0 * family,
                    "intdivinc": 200.0 * family,
                    "ssretinc": 500.0 * family,
                }
            )
    return pd.DataFrame.from_records(records)


def _raw_scf_full(summary: pd.DataFrame) -> pd.DataFrame:
    """A full-extract-shaped table deliberately in reverse join order."""

    full = summary.loc[:, ["y1", "yy1"]].copy()
    row = np.arange(1, len(full) + 1, dtype=np.float64)
    full["x2209"] = 10_000.0 + row
    full["x2309"] = np.where(row % 2 == 0, 2_000.0, -1.0)
    full["x2409"] = 0.0
    full["x7158"] = np.where(row % 3 == 0, 3_000.0, -7.0)
    full["x2219"] = 500.0  # 5 percent in SCF's rate encoding.
    full["x2319"] = 250.0
    full["x2419"] = -1.0
    full["x7170"] = 1_000.0
    return full.iloc[::-1].reset_index(drop=True)


def _write_scf_extracts(tmp_path, *, n_families: int = 8):
    summary = _raw_scf_summary(n_families)
    full = _raw_scf_full(summary)
    summary_path = tmp_path / "rscfp2022.dta"
    full_path = tmp_path / "p22i6.dta"
    summary.to_stata(summary_path, write_index=False)
    full.to_stata(full_path, write_index=False)
    return summary, full, summary_path, full_path


def _ready_donor(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    donor = pd.DataFrame({name: rng.normal(size=n) for name in SCF_WEALTH_PREDICTORS})
    donor["age"] = rng.integers(18, 90, n).astype(np.float64)
    donor["is_female"] = rng.integers(0, 2, n).astype(np.float64)
    donor["cps_race"] = rng.integers(1, 8, n).astype(np.float64)
    donor["is_married"] = rng.integers(0, 2, n).astype(np.float64)
    donor["own_children_in_household"] = rng.integers(0, 4, n).astype(np.float64)
    donor["employment_income"] = rng.gamma(2.0, 20_000.0, n)
    donor["interest_dividend_income"] = rng.gamma(1.0, 1_000.0, n)
    donor["social_security_pension_income"] = rng.gamma(1.0, 7_000.0, n)
    has_loan = rng.random(n) < 0.35
    donor["auto_loan_balance"] = np.where(has_loan, rng.gamma(2.0, 12_000.0, n), 0.0)
    donor["auto_loan_interest"] = np.where(
        has_loan, donor["auto_loan_balance"] * rng.uniform(0.02, 0.10, n), 0.0
    )
    donor[_DONOR_WEIGHT_COLUMN] = rng.uniform(500.0, 2_000.0, n)
    return donor


def _person_rows(n_households: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    records: list[dict[str, object]] = []
    person_id = 1
    for household_id in range(1, n_households + 1):
        for line, age in ((1, rng.integers(25, 75)), (2, rng.integers(0, 70))):
            records.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "PH_SEQ": household_id,
                    "A_LINENO": line,
                    "age": float(age),
                    "is_female": bool(rng.integers(0, 2)),
                    "PRDTRACE": int(rng.integers(1, 7)),
                    "PRDTHSP": int(rng.integers(0, 2)),
                    "A_MARITL": int(rng.choice([1, 2, 4, 5, 6, 7])),
                    "PEPAR1": -1,
                    "PEPAR2": -1,
                    "employment_income_before_lsr": float(rng.gamma(2.0, 12_000.0)),
                    "taxable_interest_income": float(rng.gamma(1.0, 500.0)),
                    "social_security_retirement": float(rng.gamma(1.0, 3_000.0)),
                }
            )
            person_id += 1
    return pd.DataFrame.from_records(records)


def _us_frame(
    person: pd.DataFrame,
    *,
    weights: np.ndarray | None = None,
) -> Frame:
    person = person.copy()
    household_ids = np.sort(person["person_household_id"].unique())
    n_person = len(person)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n_person) + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2_000}),
        "family": pd.DataFrame({"family_id": household_ids + 3_000}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.arange(n_person) + 4_000}),
    }
    values = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.full(len(household_ids), 1_000_000.0, dtype=np.float64)
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(values=values, kind=WeightKind.DESIGN)},
    )


def test_full_extract_coordinates_and_pin_state_are_explicit() -> None:
    assert SCF_2022_FULL_EXTRACT_URL.endswith("/scf2022s.zip")
    assert SCF_2022_FULL_EXTRACT_MEMBER == "p22i6.dta"
    pins = (
        SCF_2022_FULL_EXTRACT_ZIP_SHA256,
        SCF_2022_FULL_EXTRACT_MEMBER_SHA256,
    )
    # Never silently mix a pinned and unpinned artifact, and never substitute
    # the unrelated summary-extract hashes.  This environment cannot make the
    # one network fetch needed to fill the full-file pins; once filled, both
    # values must be sha-256 hex digests.
    assert all(pin is None for pin in pins) or all(
        isinstance(pin, str) and len(pin) == 64 and set(pin) <= set("0123456789abcdef")
        for pin in pins
    )


def test_stage_spec_declares_all_three_auto_loan_outputs_and_artifact() -> None:
    spec = us_scf_auto_loans_stage_spec()
    assert set(US_SCF_AUTO_LOAN_OUTPUT_COLUMNS) <= set(spec.outputs)
    assert set(US_SCF_AUTO_LOAN_OUTPUT_COLUMNS) <= set(spec.nonnegative_outputs)
    artifact = next(
        artifact
        for artifact in spec.artifacts
        if artifact.get("locator") == SCF_2022_FULL_EXTRACT_URL
    )
    assert artifact.get("member") == SCF_2022_FULL_EXTRACT_MEMBER
    assert artifact.get("expected_rows") == 22_975
    if SCF_2022_FULL_EXTRACT_ZIP_SHA256 is None:
        assert "pending" in artifact.get("integrity_note", "")
    else:
        assert artifact.get("sha256") == SCF_2022_FULL_EXTRACT_ZIP_SHA256
        assert artifact.get("member_sha256") == SCF_2022_FULL_EXTRACT_MEMBER_SHA256


def test_source_column_pairs_match_four_retired_vehicle_loans() -> None:
    assert SCF_AUTO_LOAN_AMOUNT_COLUMNS == ("x2209", "x2309", "x2409", "x7158")
    assert SCF_AUTO_LOAN_RATE_COLUMNS == ("x2219", "x2319", "x2419", "x7170")


def test_load_donor_joins_all_five_implicates_and_derives_targets(tmp_path) -> None:
    summary, full, summary_path, full_path = _write_scf_extracts(tmp_path, n_families=2)

    donor = load_scf_2022_auto_loan_donor(summary_path, full_path)

    assert len(donor) == 10
    assert list(donor.columns) == [
        *SCF_WEALTH_PREDICTORS,
        "auto_loan_balance",
        "auto_loan_interest",
        _DONOR_WEIGHT_COLUMN,
    ]
    # The join follows summary order even though the full extract was reversed.
    first_key = summary.loc[0, ["y1", "yy1"]].tolist()
    full_first = full.loc[
        (full["y1"] == first_key[0]) & (full["yy1"] == first_key[1])
    ].iloc[0]
    amounts = np.maximum(
        full_first.loc[list(SCF_AUTO_LOAN_AMOUNT_COLUMNS)].to_numpy(float), 0.0
    )
    rates = (
        np.maximum(
            full_first.loc[list(SCF_AUTO_LOAN_RATE_COLUMNS)].to_numpy(float), 0.0
        )
        / 10_000.0
    )
    assert donor.loc[0, "auto_loan_balance"] == pytest.approx(amounts.sum())
    assert donor.loc[0, "auto_loan_interest"] == pytest.approx(
        float(np.dot(amounts, rates))
    )
    assert (donor[["auto_loan_balance", "auto_loan_interest"]] >= 0).all().all()


def test_load_donor_rejects_an_unmatched_implicate(tmp_path) -> None:
    _, _, summary_path, full_path = _write_scf_extracts(tmp_path)
    full = pd.read_stata(full_path, convert_categoricals=False).iloc[:-1]
    full.to_stata(full_path, write_index=False)

    with pytest.raises(ValueError, match="one-to-one|unmatched"):
        load_scf_2022_auto_loan_donor(summary_path, full_path)


def test_load_donor_rejects_missing_auto_source_column(tmp_path) -> None:
    _, _, summary_path, full_path = _write_scf_extracts(tmp_path)
    full = pd.read_stata(full_path, convert_categoricals=False).drop(columns=["x7158"])
    full.to_stata(full_path, write_index=False)

    with pytest.raises(ValueError, match="missing required column"):
        load_scf_2022_auto_loan_donor(summary_path, full_path)


def test_reference_person_mask_matches_retired_scf_rules() -> None:
    # hh1 mixed-sex married couple -> male (row 1), even though female is older.
    # hh2 same-sex married couple -> older (row 2).
    # hh3 non-couple multi-adult -> oldest adult (row 5).
    # hh4 child-only -> oldest person (row 6).
    person = pd.DataFrame(
        {
            "person_household_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "age": [55, 50, 60, 45, 22, 70, 16, 11],
            "is_female": [True, False, True, True, False, True, False, True],
            "A_MARITL": [1, 1, 2, 2, 7, 7, 7, 7],
            "A_LINENO": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )

    mask = _scf_reference_person_mask(person)

    np.testing.assert_array_equal(
        mask, [False, True, True, False, False, True, True, False]
    )


def test_imputation_is_household_grain_nonnegative_and_deterministic() -> None:
    frame = _us_frame(_person_rows(80))
    donor = _ready_donor()

    a = impute_us_scf_auto_loans(frame, donor, seed=42, n_estimators=20)
    b = impute_us_scf_auto_loans(frame, donor, seed=42, n_estimators=20)

    assert list(a.columns) == ["auto_loan_balance", "auto_loan_interest"]
    assert len(a) == frame.n("household")
    assert (a >= 0).all().all()
    assert a["auto_loan_balance"].nunique() > 1
    assert a["auto_loan_interest"].nunique() > 1
    pd.testing.assert_frame_equal(a, b)


def test_qualified_interest_proxy_targets_six_million_annual_loans() -> None:
    interest = np.asarray([0.0, 1_000.0, 2_000.0])
    weights = np.asarray([2_000_000.0, 4_000_000.0, 6_000_000.0])

    qualified, share = qualified_auto_loan_interest_proxy(interest, weights)

    # Positive-loan household mass is 10m; the IRS/Treasury annual estimate is
    # 6m qualifying loans, so expected qualifying interest uses a 60% share.
    assert QUALIFIED_AUTO_LOAN_ANNUAL_ISSUANCE_TARGET == 6_000_000.0
    assert share == pytest.approx(0.6)
    np.testing.assert_allclose(qualified, [0.0, 600.0, 1_200.0])


def test_with_inputs_writes_legacy_and_obbba_household_columns() -> None:
    frame = _us_frame(_person_rows(120))

    out = with_us_scf_auto_loan_inputs(
        frame,
        seed=42,
        time_period=TIME_PERIOD,
        scf_auto_loan_donor=_ready_donor(),
        n_estimators=20,
    )

    household = out.table("household")
    for column in US_SCF_AUTO_LOAN_OUTPUT_COLUMNS:
        assert column in household.columns
        assert household[column].dtype == np.float64
        assert (household[column] >= 0).all()
    assert household["auto_loan_interest"].sum() > 0
    assert household["qualified_passenger_vehicle_loan_interest"].sum() > 0
    assert (
        household["qualified_passenger_vehicle_loan_interest"]
        <= household["auto_loan_interest"]
    ).all()


def test_with_inputs_is_idempotent_when_all_columns_have_signal() -> None:
    frame = _us_frame(_person_rows(100))
    once = with_us_scf_auto_loan_inputs(
        frame,
        seed=42,
        time_period=TIME_PERIOD,
        scf_auto_loan_donor=_ready_donor(),
        n_estimators=20,
    )
    twice = with_us_scf_auto_loan_inputs(
        once,
        seed=99,
        time_period=TIME_PERIOD,
        scf_auto_loan_donor=_ready_donor(),
        n_estimators=20,
    )

    for column in US_SCF_AUTO_LOAN_OUTPUT_COLUMNS:
        np.testing.assert_array_equal(
            once.table("household")[column], twice.table("household")[column]
        )


def test_signal_gate_passes_and_summary_reports_proxy_share() -> None:
    out = with_us_scf_auto_loan_inputs(
        _us_frame(_person_rows(160)),
        seed=4,
        time_period=TIME_PERIOD,
        scf_auto_loan_donor=_ready_donor(),
        n_estimators=20,
    )

    gate = us_scf_auto_loans_signal_gate(out)
    summary = us_scf_auto_loans_summary(out)

    assert gate.passed, gate.failures
    assert 0 < summary["auto_loan_interest_nonzero_share"] < 1
    assert 0 < summary["qualified_interest_share"] <= 1
    assert summary["auto_loan_balance_weighted_total"] > 0
    assert summary["auto_loan_interest_weighted_total"] > 0


def test_signal_gate_fails_on_missing_or_invalid_surface() -> None:
    frame = _us_frame(_person_rows(4))
    missing = us_scf_auto_loans_signal_gate(frame)
    assert not missing.passed
    assert "missing" in missing.failures[0]

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"]["auto_loan_balance"] = [0.0] * frame.n("household")
    tables["household"]["auto_loan_interest"] = [0.0] * frame.n("household")
    tables["household"]["qualified_passenger_vehicle_loan_interest"] = [1.0] * frame.n(
        "household"
    )
    invalid = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    gate = us_scf_auto_loans_signal_gate(invalid)
    assert not gate.passed
    assert any("constant" in failure for failure in gate.failures)
    assert any("exceeds auto_loan_interest" in failure for failure in gate.failures)


def test_fetch_returns_sha_matching_cached_full_extract(tmp_path) -> None:
    cached = tmp_path / SCF_2022_FULL_EXTRACT_MEMBER
    cached.write_bytes(b"pinned-full-scf")
    digest = _sha256_hexdigest(cached.read_bytes())

    result = fetch_scf_2022_full_extract(
        cache_dir=tmp_path,
        expected_member_sha256=digest,
        expected_zip_sha256=None,
    )

    assert result == cached
    assert result.read_bytes() == b"pinned-full-scf"
